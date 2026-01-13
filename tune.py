#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import builtins
import pathlib

import torch
import torch.amp
import torch.utils
import torch.utils.data
import torchvision as tv

import transformer_flow
import utils


def _load_state_dict_custom(model, ckpt):
    
    # 1. Handle Sequence Length (Interpolation)
    old_num_patches = ckpt['blocks.0.pos_embed'].shape[0]
    new_num_patches = model.num_patches
    old_grid = int(old_num_patches**0.5)
    new_grid = int(new_num_patches**0.5)
    
    filtered_dict = {}
    for k, v in ckpt.items():
        # Interpolate Position Embeddings
        if 'pos_embed' in k:
            pos_tokens = v.unsqueeze(0).transpose(1, 2).reshape(1, -1, old_grid, old_grid)
            interpolated = torch.nn.functional.interpolate(
                pos_tokens, size=(new_grid, new_grid), mode='bicubic', align_corners=False
            )
            v = interpolated.reshape(1, -1, new_num_patches).transpose(1, 2).squeeze(0)
            
        # Interpolate Prior Variance (var)
        elif k == 'var':
            var_tokens = v.unsqueeze(0).transpose(1, 2).reshape(1, -1, old_grid, old_grid)
            interpolated_var = torch.nn.functional.interpolate(
                var_tokens, size=(new_grid, new_grid), mode='bicubic'
            )
            v = interpolated_var.reshape(1, -1, new_num_patches).transpose(1, 2).squeeze(0)

        # Skip incompatible class embeddings and buffers
        if 'class_embed' in k:
            print(f"Skipping {k}: AFHQ(3) -> ImageNet(1000)")
            continue
        if 'attn_mask' in k:
            continue # Use the mask already in your 128x128 model
            
        filtered_dict[k] = v

    # 2. Load with strict=False to allow class_embed to stay initialized for ImageNet
    msg = model.load_state_dict(filtered_dict, strict=False)
    print(f"Load Result: {msg}")
    return model


def main(args):
    dist = utils.Distributed()
    utils.set_random_seed(100 + dist.rank)
    data, num_classes = utils.get_data(args.dataset, args.img_size, args.data)

    def print(*args, **kwargs):
        if dist.local_rank == 0:
            builtins.print(*args, **kwargs)

    print(f'{" Config ":-^80}')
    for k, v in sorted(vars(args).items()):
        print(f'{k:32s}: {v}')

    fid = utils.FID(reset_real_features=False, normalize=True).to('cuda')
    fid_stats_file = args.data / f'{args.dataset}_{args.img_size}_fid_stats.pth'
    if fid_stats_file.exists():
        print(f'Loading FID stats from {fid_stats_file}')
        fid.load_state_dict(torch.load(fid_stats_file, map_location='cpu', weights_only=False))
    else:
        raise FileNotFoundError(f'FID stats file "{fid_stats_file}" not found, run prepare_fid_stats.py.')
    dist.barrier()

    fixed_noise = torch.randn(
        args.num_samples // dist.world_size,
        (args.img_size // args.patch_size) ** 2,
        args.channel_size * args.patch_size**2,
    )
    if num_classes:
        fixed_y = torch.randint(num_classes, (args.num_samples // dist.world_size,))
    else:
        fixed_y = None
    data_sampler = torch.utils.data.DistributedSampler(data, num_replicas=dist.world_size, rank=dist.rank, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        data,
        sampler=data_sampler,
        batch_size=args.batch_size // dist.world_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    model = transformer_flow.Model(
        in_channels=args.channel_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        channels=args.channels,
        num_blocks=args.blocks,
        layers_per_block=args.layers_per_block,
        nvp=args.nvp,
        num_classes=num_classes,
    ).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=args.lr, weight_decay=1e-4)
    lr_schedule = utils.CosineLRSchedule(optimizer, len(data_loader), args.epochs * len(data_loader), 1e-6, args.lr)
    scaler = torch.amp.GradScaler()
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
        model = _load_state_dict_custom(model, ckpt)
        # ckpt = torch.load(args.resume.replace('_model_', '_opt_'), map_location='cpu')
        # optimizer.load_state_dict(ckpt['optimizer'])
        # lr_schedule.load_state_dict(ckpt['lr_schedule'])
        del ckpt
        print(f'Loaded checkpoint {args.resume}')

    if dist.distributed:
        model_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.local_rank])
    else:
        model_ddp = model

    if args.noise_type == 'gaussian':
        model_name = f'{args.patch_size}_{args.channels}_{args.blocks}_{args.layers_per_block}_{args.noise_std:.2f}'
    else:
        model_name = f'{args.patch_size}_{args.channels}_{args.blocks}_{args.layers_per_block}_uniform'
    sample_dir: pathlib.Path = args.logdir / f'{args.dataset}_samples_{model_name}'
    model_ckpt_file = args.logdir / f'{args.dataset}_model_{model_name}.pth'
    opt_ckpt_file = args.logdir / f'{args.dataset}_opt_{model_name}.pth'
    if dist.local_rank == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    enable_amp = args.noise_type == 'gaussian'
    def compute_loss(x, y):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=enable_amp):
            z, outputs, logdets = model_ddp(x, y)
            loss = model.get_loss(z, logdets)
            return loss, (z, outputs, logdets)

    if args.compile:
        compute_loss = torch.compile(compute_loss, fullgraph=False, backend='inductor', mode='max-autotune')
        dist.barrier()

    print(f'{" Training ":-^80}')
    accum_steps = args.accum_steps
    for epoch in range(args.epochs):
        metrics = utils.Metrics()
        for step, (x, y) in enumerate(data_loader):
            x = x.cuda(non_blocking=True)
            if args.noise_type == 'gaussian':
                eps = args.noise_std * torch.randn_like(x)
                x = x + eps
            elif args.noise_type == 'uniform':
                x_int = (x + 1) * (255 / 2)
                x = (x_int + torch.rand_like(x_int)) / 256
                x = x * 2 - 1
            if num_classes:
                y = y.cuda(non_blocking=True)
                mask = (torch.rand(y.size(0), device='cuda') < args.drop_label).int()
                # we use -1 to denote dropped classes
                y = (1 - mask) * y - mask
            else:
                y = None
            optimizer.zero_grad()
            loss, (z, outputs, logdets) = compute_loss(x, y)
            if not args.nvp:
                model.update_prior(dist.gather_concat(z.detach().square().mean(dim=0, keepdim=True).sqrt()))
            dist.barrier()
            loss = loss / accum_steps
            scaler.scale(loss).backward()
            if ((step + 1) % accum_steps == 0) or (step + 1 == len(data_loader)):
                scaler.step(optimizer)
                scaler.update()
                current_lr = lr_schedule.step()
                optimizer.zero_grad()
                true_loss = loss * accum_steps
                metrics.update({'loss': true_loss, 'loss/mse(z)': 0.5 * (z.detach() ** 2).mean(), 'loss/log(|det|)': logdets.mean()})
                if args.dry_run and step + 1 >= 100:
                    break

        metrics_dict = {'lr': current_lr, **metrics.compute(dist)}  # pyright: ignore[reportPossiblyUnboundVariable]
        if dist.local_rank == 0:
            metrics.print(metrics_dict, epoch + 1)
            print('\tLayer norm', ' '.join([f'{z.pow(2).mean():.4f}' for z in outputs]))  # pyright: ignore[reportPossiblyUnboundVariable]
            torch.save(model.state_dict(), model_ckpt_file)
            torch.save({'optimizer': optimizer.state_dict(), 'lr_schedule': lr_schedule.state_dict()}, opt_ckpt_file)
        dist.barrier()

        if (epoch + 1) % args.sample_freq == 0:
            for i in range(args.num_samples // args.sample_batch_size):
                b = args.sample_batch_size // dist.world_size
                noise = fixed_noise[i * b : (i + 1) * b].to('cuda')
                y = None if fixed_y is None else fixed_y[i * b : (i + 1) * b].to('cuda')
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        samples = model.reverse(noise, y, guidance=args.cfg)
                        assert isinstance(samples, torch.Tensor)
                    samples = dist.gather_concat(samples)
                    fid.update(0.5 * (samples.clip(min=-1, max=1) + 1), real=False)
                if args.dry_run:
                    break
            fid_score = fid.compute().item()
            fid.reset()

            if dist.local_rank == 0:
                utils.Metrics.print({'fid': fid_score}, epoch + 1)
                tv.utils.save_image(
                    samples,  # pyright: ignore[reportPossiblyUnboundVariable]
                    sample_dir / f'samples_{epoch+1:03d}.png', 
                    normalize=True, 
                    nrow=16
                )
            dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data', type=pathlib.Path, help='Path for training data')
    parser.add_argument('--logdir', default='runs', type=pathlib.Path, help='Path for artifacts')
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'imagenet64', 'afhq'], help='Name of dataset')
    parser.add_argument('--img_size', default=64, type=int, help='Image size')
    parser.add_argument('--channel_size', default=3, type=int, help='Image channel size')

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

    parser.add_argument('--nvp', default=True, action=argparse.BooleanOptionalAction, help='Whether to use the non volume preserving version')
    parser.add_argument(
        '--compile', default=False, action=argparse.BooleanOptionalAction, help='Whether to use torch.compile, expect the first epoch to be slow when enabled'
    )
    parser.add_argument(
        '--dry_run', default=False, action=argparse.BooleanOptionalAction, help='Dry run for quick tests'
    )
    args = parser.parse_args()

    main(args)
