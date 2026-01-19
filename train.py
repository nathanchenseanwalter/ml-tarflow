#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import json
import logging
import pathlib
import sys
import time
import warnings

import torch
import torch.amp
import torch.utils
import torch.utils.data
import torchvision as tv

import transformer_flow
import utils


def main(args):

    # distributed training
    dist = utils.Distributed()
    utils.set_random_seed(100 + dist.rank)
    
    # device setup
    device = torch.device(f'cuda:{dist.local_rank}' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    # logging - use stdout instead of stderr
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO if dist.rank == 0 else logging.WARNING,
        stream=sys.stdout,
        force=True,  # override any existing handlers
    )
    logger = logging.getLogger(__name__)
    
    # only show warnings on rank 0 to avoid repeated messages
    if dist.rank != 0:
        warnings.filterwarnings('ignore')

    # print config
    logger.info('Config:')
    for k, v in sorted(vars(args).items()):
        logger.info(f'{k:32s}: {v}')
    logger.info('=' * 80 + '\n')

    # data setup
    data, num_classes = utils.get_data(args.dataset, args.img_size, args.data)

    # setup fid
    fid = utils.FID(reset_real_features=False, normalize=True).to(device)
    fid_stats_file = args.data / f'{args.dataset}_{args.img_size}_fid_stats.pth'
    if fid_stats_file.exists():
        logger.info(f'Loading FID stats from {fid_stats_file}')
        fid.load_state_dict(torch.load(fid_stats_file, map_location='cpu', weights_only=False))
    else:
        raise FileNotFoundError(f'FID stats file "{fid_stats_file}" not found, run prepare_fid_stats.py.')

    # barrier
    dist.barrier()

    # fixed noise and labels
    fixed_noise = torch.randn(
        args.num_samples // dist.world_size,
        (args.img_size // args.patch_size) ** 2,
        args.channel_size * args.patch_size**2,
    )
    if num_classes:
        fixed_y = torch.randint(num_classes, (args.num_samples // dist.world_size,))
    else:
        fixed_y = None

    # data loading
    data_sampler = torch.utils.data.DistributedSampler(
        data, 
        num_replicas=dist.world_size, 
        rank=dist.rank, 
        shuffle=True,
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        sampler=data_sampler,
        batch_size=args.batch_size // dist.world_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # model setup
    model = transformer_flow.Model(
        in_channels=args.channel_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        channels=args.channels,
        num_blocks=args.blocks,
        layers_per_block=args.layers_per_block,
        nvp=args.nvp,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        betas=(0.9, 0.95), 
        lr=args.lr, 
        weight_decay=1e-4,
    )
    lr_schedule = utils.CosineLRSchedule(
        optimizer, 
        len(data_loader), 
        args.epochs * len(data_loader), 
        1e-6, 
        args.lr,
    )
    
    # precision setup
    use_fp16 = args.precision == 'fp16'
    use_amp = args.noise_type == 'gaussian' and args.precision != 'fp32'
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=use_fp16) # pyright: ignore

    # tracking best model
    best_fid = float('inf')
    start_epoch = 0

    # file naming
    if args.noise_type == 'gaussian':
        model_name = f'{args.patch_size}_{args.channels}_{args.blocks}_{args.layers_per_block}_{args.noise_std:.2f}'
    else:
        model_name = f'{args.patch_size}_{args.channels}_{args.blocks}_{args.layers_per_block}_uniform'
    sample_dir: pathlib.Path = args.logdir / f'{args.dataset}_samples_{model_name}'
    model_ckpt_file = args.logdir / f'{args.dataset}_model_{model_name}.pth'
    opt_ckpt_file = args.logdir / f'{args.dataset}_opt_{model_name}.pth'
    best_model_ckpt_file = args.logdir / f'{args.dataset}_model_{model_name}_best.pth'
    config_file = args.logdir / f'{args.dataset}_config_{model_name}.json'
    
    if dist.rank == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)
        args.logdir.mkdir(parents=True, exist_ok=True)
        
        # save config
        with open(config_file, 'w') as f:
            json.dump({k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()}, f, indent=2)
        logger.info(f'Saved config to {config_file}')

    # load checkpoint
    if args.resume:
        resume_path = pathlib.Path(args.resume)
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location='cpu', weights_only=True)
            model.load_state_dict(ckpt)
            logger.info(f'Loaded model checkpoint from {resume_path}')
            
            opt_ckpt_path = pathlib.Path(str(resume_path).replace('_model_', '_opt_'))
            if opt_ckpt_path.exists():
                ckpt = torch.load(opt_ckpt_path, map_location='cpu', weights_only=True)
                optimizer.load_state_dict(ckpt['optimizer'])
                lr_schedule.load_state_dict(ckpt['lr_schedule'])
                start_epoch = ckpt.get('epoch', 0)
                best_fid = ckpt.get('best_fid', float('inf'))
                logger.info(f'Loaded optimizer checkpoint from {opt_ckpt_path} (epoch {start_epoch}, best_fid {best_fid:.2f})')
            else:
                logger.warning(f'Optimizer checkpoint {opt_ckpt_path} not found, starting optimizer from scratch')
            del ckpt
        else:
            logger.warning(f'Checkpoint {resume_path} not found, starting from scratch')

    # model ddp
    if dist.distributed:
        model_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.local_rank])
    else:
        model_ddp = model

    # loss setup
    def compute_loss(x, y):
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            z, outputs, logdets = model_ddp(x, y)
            loss = model.get_loss(z, logdets)
            return loss, (z, outputs, logdets)

    # compile model
    if args.compile:
        compute_loss = torch.compile(
            compute_loss, 
            fullgraph=False, 
            backend='inductor', 
            mode='max-autotune',
        )
        dist.barrier()

    # progress interval
    progress_interval = max(1, len(data_loader) // 5)  # Update every 20%

    # training loop
    logger.info(f'{" Training ":-^80}')
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        metrics = utils.Metrics()
        
        data_sampler.set_epoch(epoch)

        # step through data loader
        for step, (x, y) in enumerate(data_loader):

            # progress tracking
            if step % progress_interval == 0 or step == len(data_loader) - 1:
                progress_pct = 100 * step / len(data_loader)
                elapsed = time.time() - epoch_start_time
                eta = (elapsed / max(step, 1)) * (len(data_loader) - step)
                logger.info(f'Epoch {epoch+1}/{args.epochs} - {progress_pct:.1f}% ({step}/{len(data_loader)}) - ETA: {eta:.0f}s')
            
            # data
            x = x.to(device, non_blocking=True)
            if args.noise_type == 'gaussian':
                eps = args.noise_std * torch.randn_like(x)
                x = x + eps
            elif args.noise_type == 'uniform':
                x_int = (x + 1) * (255 / 2)
                x = (x_int + torch.rand_like(x_int)) / 256
                x = x * 2 - 1
            if num_classes:
                y = y.to(device, non_blocking=True)
                mask = (torch.rand(y.size(0), device=device) < args.drop_label).int()
                # we use -1 to denote dropped classes
                y = (1 - mask) * y - mask
            else:
                y = None
            
            # train step
            optimizer.zero_grad()
            loss, (z, outputs, logdets) = compute_loss(x, y)
            if not args.nvp:
                model.update_prior(dist.gather_concat(z.detach().square().mean(dim=0, keepdim=True).sqrt()))
            dist.barrier()
            loss = loss / args.accum_steps
            scaler.scale(loss).backward()

            # gradient update
            if ((step + 1) % args.accum_steps == 0) or (step + 1 == len(data_loader)):

                # gradient clipping
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                current_lr = lr_schedule.step()
                optimizer.zero_grad()
                true_loss = loss * args.accum_steps
                metrics.update(
                    {
                        'loss': true_loss, 
                        'loss/mse(z)': 0.5 * (z.detach() ** 2).mean(), 
                        'loss/log(|det|)': logdets.mean(),
                    }
                )
                if args.dry_run and step + 1 >= 100:
                    break

        # metrics
        epoch_time = time.time() - epoch_start_time
        metrics_dict = {'lr': current_lr, 'epoch_time': epoch_time, **metrics.compute(dist)}  # pyright: ignore[reportPossiblyUnboundVariable]
        if dist.rank == 0:
            metrics.print(metrics_dict, epoch + 1)
            logger.info('Layer norm: %s', ' '.join([f'{z.pow(2).mean():.4f}' for z in outputs]))  # pyright: ignore[reportPossiblyUnboundVariable]
            torch.save(model.state_dict(), model_ckpt_file)
            torch.save({
                'optimizer': optimizer.state_dict(), 
                'lr_schedule': lr_schedule.state_dict(),
                'epoch': epoch + 1,
                'best_fid': best_fid,
            }, opt_ckpt_file)
        dist.barrier()

        # sampling
        if (epoch + 1) % args.sample_freq == 0:
            for i in range(args.num_samples // args.sample_batch_size):
                b = args.sample_batch_size // dist.world_size
                noise = fixed_noise[i * b : (i + 1) * b].to(device)
                y = None if fixed_y is None else fixed_y[i * b : (i + 1) * b].to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                        samples = model.reverse(noise, y, guidance=args.cfg)
                        assert isinstance(samples, torch.Tensor)
                    samples = dist.gather_concat(samples)
                    fid.update(0.5 * (samples.clip(min=-1, max=1) + 1), real=False)
                if args.dry_run:
                    break
            fid_score = fid.compute().item()
            fid.reset()

            # free up memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # save best model
            if dist.rank == 0:
                utils.Metrics.print({'fid': fid_score}, epoch + 1)
                tv.utils.save_image(
                    samples,  # pyright: ignore[reportPossiblyUnboundVariable]
                    sample_dir / f'samples_{epoch+1:03d}.png', 
                    normalize=True, 
                    nrow=16
                )
                # save best model
                if fid_score < best_fid:
                    best_fid = fid_score
                    torch.save(model.state_dict(), best_model_ckpt_file)
                    # also save optimizer state for best model
                    best_opt_ckpt_file = args.logdir / f'{args.dataset}_opt_{model_name}.pth'
                    torch.save({
                        'optimizer': optimizer.state_dict(), 
                        'lr_schedule': lr_schedule.state_dict(),
                        'epoch': epoch + 1,
                        'best_fid': best_fid,
                    }, best_opt_ckpt_file)
                    logger.info(f'New best FID: {best_fid:.2f} - saved to {best_model_ckpt_file}')
            dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data', type=pathlib.Path, help='Path for training data')
    parser.add_argument('--logdir', default='runs', type=pathlib.Path, help='Path for artifacts')
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'imagenet64', 'afhq'], help='Name of dataset')
    parser.add_argument('--img_size', default=64, type=int, help='Image size')
    parser.add_argument('--channel_size', default=3, type=int, help='Image channel size')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for data loading')

    parser.add_argument('--patch_size', default=4, type=int, help='Patch size for the model')
    parser.add_argument('--channels', default=512, type=int, help='Model width')
    parser.add_argument('--blocks', default=4, type=int, help='Number of autoregressive flow blocks')
    parser.add_argument('--layers_per_block', default=8, type=int, help='Depth per flow block')
    parser.add_argument('--noise_std', default=0.05, type=float, help='Input noise standard deviation')
    parser.add_argument('--noise_type', default='gaussian', choices=['gaussian', 'uniform'], type=str)
    parser.add_argument('--cfg', default=0, type=float, help='Guidance weight for sampling, 0 is no guidance')

    parser.add_argument('--batch_size', default=128, type=int, help='Training batch size across all devices')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Maximum learning rate')
    parser.add_argument('--drop_label', default=0, type=float, help='Ratio for random label drop in conditional mode')
    parser.add_argument('--sample_freq', default=1, type=int, help='Frequency of sampling in terms of epochs')
    parser.add_argument('--num_samples', default=4096, type=int, help='Number of sampels to draw')
    parser.add_argument('--sample_batch_size', default=256, type=int, help='Batch size for drawing samples')
    parser.add_argument('--resume', default='', type=str, help='path for checkpoint to resume training from')
    parser.add_argument('--accum_steps', default=1, type=int, help='Number of steps to accumulate gradients')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='Gradient clipping max norm (0 to disable)')
    parser.add_argument('--precision', default='bf16', choices=['fp32', 'fp16', 'bf16'], help='Training precision')

    parser.add_argument('--nvp', default=True, action=argparse.BooleanOptionalAction, help='Whether to use the non volume preserving version')
    parser.add_argument(
        '--compile', default=False, action=argparse.BooleanOptionalAction, help='Whether to use torch.compile, expect the first epoch to be slow when enabled'
    )
    parser.add_argument(
        '--dry_run', default=False, action=argparse.BooleanOptionalAction, help='Dry run for quick tests'
    )
    args = parser.parse_args()

    main(args)
