import argparse, json, os, hashlib, math, time
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import transformer_flow as tf


# =========================
# Utilities
# =========================

def seed_all(seed: int = 0):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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
):
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
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd, strict=True)


# =========================
# Oracle math
# =========================

@torch.no_grad()
def std_normal_logprob_patches(z: torch.Tensor) -> torch.Tensor:
    # z: [B, T, Cp]
    B, T, Cp = z.shape
    d = T * Cp
    const = -0.5 * d * math.log(2.0 * math.pi)
    quad = -0.5 * (z.reshape(B, -1) ** 2).sum(dim=1)
    return z.new_tensor(const) + quad

@torch.no_grad()
def log_p_x_given_y(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z, _, logdet = model(x, y)
    return std_normal_logprob_patches(z) + logdet

@torch.no_grad()
def p_y_given_x(model: torch.nn.Module, x: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
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
        lp = log_p_x_given_y(model, x, yb)   # [B]
        logps.append(lp)

        if (y + 1) % 50 == 0:
            print(f"    [posterior] done {y+1}/{C} classes", flush=True)

    logps = torch.stack(logps, dim=1)        # [B, C]
    logps = logps + torch.log(pi + 1e-12)

    post = torch.softmax(logps, dim=1)
    return post



# =========================
# Oracle rejection sampling
# =========================

@torch.no_grad()
def sample_conditional_unbiased(
    model,
    per_class,
    class_index,
    img_size,
    patch_size,
    channel_size,
    pi,
    batch,
    device,
    max_attempts_per_needed=20,
    print_every=1,
):
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

        # Sample from p(x|y)
        xb = model.reverse(z0, yb)

        # [-1,1] → [0,1] for saving
        xb_img = (xb * 0.5 + 0.5).clamp(0.0, 1.0)

        # Back to model space for likelihood
        post = p_y_given_x(model, xb_img * 2 - 1, pi)
        top1 = post.argmax(dim=1)

        keep = xb_img[top1 == class_index]
        if keep.numel() > 0:
            accepted.append(keep[:needed])
            needed -= keep.size(0)

        attempts += 1

        if attempts % print_every == 0:
            accepted_so_far = per_class - needed
            acc_rate = accepted_so_far / attempts
            elapsed = (time.time() - t_start) / 60.0
            print(
                f"[class {class_index:04d}] "
                f"attempt {attempts}/{cap} | "
                f"accepted {accepted_so_far}/{per_class} | "
                f"acc_rate={acc_rate:.3f} | "
                f"elapsed={elapsed:.1f} min",
                flush=True,
            )

    if needed > 0:
        raise RuntimeError(
            f"[class {class_index}] FAILED: accepted {per_class-needed}/{per_class} "
            f"in {attempts} attempts"
        )

    return torch.cat(accepted, dim=0)


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser("ImageNet-64 EXACT oracle benchmark (VERY SLOW)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--per_class", type=int, default=10)

    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--channel_size", type=int, default=3)
    ap.add_argument("--patch_size", type=int, required=True)
    ap.add_argument("--channels", type=int, required=True)
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--layers_per_block", type=int, required=True)
    ap.add_argument("--nvp", action="store_true")

    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_attempts_per_needed", type=int, default=20)

    args = ap.parse_args()
    seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    num_classes = 1000
    pi = torch.full((num_classes,), 1.0 / num_classes, device=device)

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
    load_ckpt(model, args.ckpt, device)

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    print(f"[benchmark] ImageNet-64 oracle benchmark")
    print(f"[benchmark] per_class={args.per_class}, num_classes=1000")

    t_global = time.time()

    for yi in range(num_classes):
        t_class = time.time()
        print(f"\n===== CLASS {yi:04d} START =====", flush=True)

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
            print_every=1,
        )

        cls_dir = outdir / f"class_{yi:04d}"
        ensure_dir(cls_dir)
        for i in range(imgs.size(0)):
            to_pil(imgs[i]).save(cls_dir / f"{i:03d}.png")

        dt_class = (time.time() - t_class) / 60.0
        dt_total = (time.time() - t_global) / 60.0

        print(
            f"===== CLASS {yi:04d} DONE | "
            f"class_time={dt_class:.1f} min | "
            f"total_elapsed={dt_total:.1f} min =====",
            flush=True,
        )

    print("[benchmark] DONE")


if __name__ == "__main__":
    main()