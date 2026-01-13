import argparse
import hashlib
import json
import logging
import math
import random
import time
from pathlib import Path

from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import transformer_flow as tf


# =========================
# Utilities
# =========================

def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_pil(img: torch.Tensor) -> Image.Image:
    return transforms.ToPILImage()(img.cpu())


# =========================
# Model
# =========================

def build_model(
    img_size: int,
    channel_size: int,
    patch_size: int,
    channels: int,
    blocks: int,
    layers_per_block: int,
    nvp: bool,
    num_classes: int,
    device: torch.device,
) -> tf.Model:
    model = tf.Model(
        in_channels=channel_size,
        img_size=img_size,
        patch_size=patch_size,
        channels=channels,
        num_blocks=blocks,
        layers_per_block=layers_per_block,
        nvp=nvp,
        num_classes=num_classes,
    ).to(device=device, dtype=torch.float32)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_ckpt(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd, strict=True)
    model.eval()


# =========================
# Oracle math
# =========================

@torch.no_grad()
def std_normal_logprob_patches(z: torch.Tensor) -> torch.Tensor:
    """Compute log probability under standard normal for patched tensor."""
    B, T, Cp = z.shape
    d = T * Cp
    const = -0.5 * d * math.log(2.0 * math.pi)
    quad = -0.5 * (z.reshape(B, -1) ** 2).sum(dim=1)
    return z.new_tensor(const) + quad


@torch.no_grad()
def log_p_x_given_y(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute log p(x|y) using the flow model."""
    z, _, logdet = model(x, y)
    return std_normal_logprob_patches(z) + logdet


@torch.no_grad()
def p_y_given_x(
    model: torch.nn.Module, 
    x: torch.Tensor, 
    pi: torch.Tensor,
    logger: Optional[logging.Logger] = None,
    log_every: int = 50,
) -> torch.Tensor:
    """
    Memory-safe exact posterior.
    Computes log p(x|y) one class at a time.
    VERY SLOW but avoids OOM.
    """
    device = x.device
    pi = pi.to(device)
    C = pi.numel()
    B = x.size(0)

    logps = []

    for y in range(C):
        yb = torch.full((B,), y, device=device, dtype=torch.long)
        lp = log_p_x_given_y(model, x, yb)
        logps.append(lp)

        if logger and (y + 1) % log_every == 0:
            logger.debug(f"[posterior] done {y+1}/{C} classes")

    logps = torch.stack(logps, dim=1)  # [B, C]
    logps = logps + torch.log(pi + 1e-12)
    post = torch.softmax(logps, dim=1)
    return post


# =========================
# Oracle rejection sampling
# =========================

@torch.no_grad()
def sample_conditional_unbiased(
    model: tf.Model,
    per_class: int,
    class_index: int,
    img_size: int,
    patch_size: int,
    channel_size: int,
    pi: torch.Tensor,
    batch: int,
    device: torch.device,
    max_attempts_per_needed: int = 20,
    logger: Optional[logging.Logger] = None,
) -> torch.Tensor:
    """Sample from p(x|y) with rejection sampling for oracle benchmark."""
    T = (img_size // patch_size) ** 2
    Cp = channel_size * (patch_size ** 2)

    accepted = []
    needed = per_class
    attempts = 0
    cap = max_attempts_per_needed * per_class
    t_start = time.time()

    while needed > 0 and attempts < cap:
        b = min(batch, needed * 2)

        z0 = torch.randn(b, T, Cp, device=device)
        yb = torch.full((b,), class_index, device=device, dtype=torch.long)

        # sample from p(x|y)
        xb = model.reverse(z0, yb)

        # [-1,1] → [0,1] for saving
        xb_img = (xb * 0.5 + 0.5).clamp(0.0, 1.0)  # type: ignore

        # back to model space for likelihood
        post = p_y_given_x(model, xb_img * 2 - 1, pi, logger=logger)
        top1 = post.argmax(dim=1)

        keep = xb_img[top1 == class_index]
        if keep.numel() > 0:
            accepted.append(keep[:needed])
            needed -= keep.size(0)

        attempts += 1

        if logger:
            accepted_so_far = per_class - needed
            acc_rate = accepted_so_far / attempts if attempts > 0 else 0
            elapsed = (time.time() - t_start) / 60.0
            logger.info(
                f"[class {class_index:04d}] attempt {attempts}/{cap} | "
                f"accepted {accepted_so_far}/{per_class} | "
                f"acc_rate={acc_rate:.3f} | elapsed={elapsed:.1f} min"
            )

    if needed > 0:
        raise RuntimeError(
            f"[class {class_index}] FAILED: accepted {per_class-needed}/{per_class} "
            f"in {attempts} attempts"
        )

    return torch.cat(accepted, dim=0)


# =========================
# Progress tracking
# =========================

def load_progress(progress_file: Path) -> set[int]:
    """Load completed class indices from progress file."""
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return set(json.load(f).get('completed_classes', []))
    return set()


def save_progress(progress_file: Path, completed: set[int], total_time: float):
    """Save progress to file for resumption."""
    with open(progress_file, 'w') as f:
        json.dump({
            'completed_classes': sorted(completed),
            'total_time_minutes': total_time,
        }, f, indent=2)


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser("ImageNet-128 EXACT oracle benchmark (VERY SLOW)")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    ap.add_argument("--outdir", required=True, help="Output directory for samples")
    ap.add_argument("--per_class", type=int, default=10, help="Samples per class")

    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--channel_size", type=int, default=3)
    ap.add_argument("--patch_size", type=int, required=True)
    ap.add_argument("--channels", type=int, required=True)
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--layers_per_block", type=int, required=True)
    ap.add_argument("--nvp", action="store_true")

    ap.add_argument("--batch", type=int, default=1, help="Batch size for sampling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_attempts_per_needed", type=int, default=20)
    ap.add_argument("--resume", action="store_true", help="Resume from previous run")
    ap.add_argument("--start_class", type=int, default=0, help="Starting class index")
    ap.add_argument("--end_class", type=int, default=1000, help="Ending class index (exclusive)")

    args = ap.parse_args()

    # logging setup
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    seed_all(args.seed)

    # device setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    num_classes = 1000
    pi = torch.full((num_classes,), 1.0 / num_classes, device=device)

    # build and load model
    logger.info("Building model...")
    model = build_model(
        args.img_size,
        args.channel_size,
        args.patch_size,
        args.channels,
        args.blocks,
        args.layers_per_block,
        args.nvp,
        num_classes,
        device,
    )
    logger.info(f"Loading checkpoint from {args.ckpt}")
    load_ckpt(model, args.ckpt, device)

    # output directory setup
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    progress_file = outdir / "progress.json"
    config_file = outdir / "config.json"

    # save config
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # load progress if resuming
    completed_classes = load_progress(progress_file) if args.resume else set()
    if completed_classes:
        logger.info(f"Resuming: {len(completed_classes)} classes already completed")

    logger.info(f"ImageNet-128 oracle benchmark")
    logger.info(f"per_class={args.per_class}, classes={args.start_class}-{args.end_class}")

    t_global = time.time()

    for yi in range(args.start_class, args.end_class):
        # skip completed classes
        if yi in completed_classes:
            logger.info(f"Skipping class {yi:04d} (already completed)")
            continue

        t_class = time.time()
        logger.info(f"===== CLASS {yi:04d} START =====")

        try:
            imgs = sample_conditional_unbiased(
                model=model,
                per_class=args.per_class,
                class_index=yi,
                img_size=args.img_size,
                patch_size=args.patch_size,
                channel_size=args.channel_size,
                pi=pi,
                batch=args.batch,
                device=device,
                max_attempts_per_needed=args.max_attempts_per_needed,
                logger=logger,
            )

            # save images
            cls_dir = outdir / f"class_{yi:04d}"
            ensure_dir(cls_dir)
            for i in range(imgs.size(0)):
                to_pil(imgs[i]).save(cls_dir / f"{i:03d}.png")

            # update progress
            completed_classes.add(yi)
            total_elapsed = (time.time() - t_global) / 60.0
            save_progress(progress_file, completed_classes, total_elapsed)

            dt_class = (time.time() - t_class) / 60.0
            remaining = args.end_class - yi - 1
            eta = (total_elapsed / max(len(completed_classes), 1)) * remaining

            logger.info(
                f"===== CLASS {yi:04d} DONE | "
                f"class_time={dt_class:.1f} min | "
                f"total_elapsed={total_elapsed:.1f} min | "
                f"ETA={eta:.1f} min ====="
            )

        except RuntimeError as e:
            logger.error(f"Failed on class {yi}: {e}")
            continue

        # free memory periodically
        if device.type == 'cuda' and (yi + 1) % 10 == 0:
            torch.cuda.empty_cache()

    total_time = (time.time() - t_global) / 60.0
    logger.info(f"DONE - Total time: {total_time:.1f} minutes")
    logger.info(f"Completed {len(completed_classes)} classes")


if __name__ == "__main__":
    main()
