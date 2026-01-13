import argparse, json
from pathlib import Path
from typing import List, Optional
import torch
from torchvision import transforms
from PIL import Image
import transformer_flow as tf  


def seed_all(seed: int = 0):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def preprocess(img_size: int):
    return transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),                       
    ])

def load_images(paths: List[str], img_size: int, device: torch.device) -> torch.Tensor:
    tfm = preprocess(img_size)
    xs = []
    for p in paths:
        with Image.open(p).convert("RGB") as im:
            xs.append(tfm(im))
    x = torch.stack(xs, 0).to(device=device, dtype=torch.float32) 
    return x

def load_prior(classes: List[str], prior_json: Optional[str]) -> torch.Tensor:
    if prior_json and Path(prior_json).exists():
        meta = json.loads(Path(prior_json).read_text())
        order = meta.get("order", classes)
        pi_dict = meta.get("pi", None)
        if pi_dict is not None and order == classes:
            pi = torch.tensor([pi_dict[c] for c in classes], dtype=torch.float32)
            s = pi.sum()
            if s > 0: pi = pi / s
            return pi
    C = len(classes)
    return torch.full((C,), 1.0 / C, dtype=torch.float32)


def build_model(img_size:int, channel_size:int, patch_size:int,
                channels:int, blocks:int, layers_per_block:int,
                nvp:bool, num_classes:int, device:torch.device) -> torch.nn.Module:
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

def std_normal_logprob_patches(z_patches: torch.Tensor) -> torch.Tensor:
    """
    z_patches: [B, T, C'] from Model.forward
    returns: [B] log p(z) under iid N(0,1) over T*C' dims
    """
    if z_patches.dim() != 3:
        raise RuntimeError(f"Expected z in patch space [B,T,C'], got {tuple(z_patches.shape)}")
    B, T, Cp = z_patches.shape
    d = T * Cp
    const = -0.5 * d * torch.log(torch.tensor(2.0 * torch.pi, dtype=z_patches.dtype, device=z_patches.device))
    quad = -0.5 * (z_patches.reshape(B, -1) ** 2).sum(dim=1)
    return const + quad  # [B]


@torch.no_grad()
def log_p_x_given_y(
    model: torch.nn.Module, 
    x: torch.Tensor, 
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Computes log p(x|y) using the model's forward in patch space:
    forward(x,y) -> (z_patches [B,T,C'], outputs_list, logdet_sum [B])
    """
    out = model(x, y)
    if not (isinstance(out, (tuple, list)) and len(out) == 3):
        raise RuntimeError(
            "Expected Model.forward(x,y) to return (z_patches, outputs_list, logdet_sum). "
            f"Got type/len: {type(out)} / {len(out) if isinstance(out,(tuple,list)) else 'n/a'}"
        )
    z_patches, _outputs_list, logdet_sum = out
    if z_patches.dim() != 3 or logdet_sum.dim() != 1 or logdet_sum.size(0) != x.size(0):
        raise RuntimeError(
            f"Unexpected shapes: z_patches={tuple(z_patches.shape)} (want [B,T,C']), "
            f"logdet_sum={tuple(logdet_sum.shape)} (want [B]), x={tuple(x.shape)}"
        )
    log_pz = std_normal_logprob_patches(z_patches) 
    return log_pz + logdet_sum       

@torch.no_grad()
def log_p_x(model: torch.nn.Module, x: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    device = x.device
    pi = pi.to(device=device, dtype=torch.float32)
    C = pi.numel(); B = x.size(0)
    ys = torch.arange(C, device=device, dtype=torch.long)
    x_rep = x.unsqueeze(0).expand(C, B, *x.shape[1:]).reshape(C * B, *x.shape[1:])
    y_rep = ys.unsqueeze(1).expand(C, B).reshape(-1)
    lpy = log_p_x_given_y(model, x_rep, y_rep).view(C, B)    # [C,B]
    log_pi = torch.log(pi + 1e-12).unsqueeze(1)              # [C,1]
    m = torch.max(log_pi + lpy, dim=0).values                # [B]
    log_px = m + torch.log(torch.sum(torch.exp(log_pi + lpy - m.unsqueeze(0)), dim=0) + 1e-12)
    return log_px                                            # [B]

@torch.no_grad()
def p_y_given_x(model: torch.nn.Module, x: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    device = x.device
    pi = pi.to(device=device, dtype=torch.float32)
    C = pi.numel(); B = x.size(0)
    ys = torch.arange(C, device=device, dtype=torch.long)
    x_rep = x.unsqueeze(0).expand(C, B, *x.shape[1:]).reshape(C * B, *x.shape[1:])
    y_rep = ys.unsqueeze(1).expand(C, B).reshape(-1)
    lpy = log_p_x_given_y(model, x_rep, y_rep).view(C, B)    # [C,B]
    log_pi = torch.log(pi + 1e-12).unsqueeze(1)              # [C,1]
    log_num = log_pi + lpy                                    # [C,B]
    log_den = torch.logsumexp(log_num, dim=0, keepdim=True)   # [1,B]
    post = torch.exp(log_num - log_den).transpose(0,1).contiguous()  # [B,C]
    return post


def load_classes_from_file(path: str) -> List[str]:
    """Load class IDs from a text file, one per line."""
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser("ImageNet Oracle (TarFlow)")
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--channel_size", type=int, default=3)
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--channels", type=int, default=1024)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--layers_per_block", type=int, default=8)
    ap.add_argument("--nvp", action="store_true", default=True)
    ap.add_argument("--classes", nargs="+", default=None,
                    help="Class list; if omitted, reads from --classes_file")
    ap.add_argument("--classes_file", type=str, default="data/imagenet_classes.txt",
                    help="File with one class ID per line (used if --classes not provided)")
    ap.add_argument("--prior_json", type=str, default="data/imagenet_prior.json")
    ap.add_argument("--device", type=str, default="cuda")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("post");    sp.add_argument("images", nargs="+")
    sp2 = sub.add_parser("logpxy"); sp2.add_argument("--y", type=int, required=True); sp2.add_argument("images", nargs="+")
    sp3 = sub.add_parser("logpx");  sp3.add_argument("images", nargs="+")

    args = ap.parse_args()
    seed_all(0)

    # Load classes from file if not provided on command line
    if args.classes is None:
        args.classes = load_classes_from_file(args.classes_file)
        print(f"[info] Loaded {len(args.classes)} classes from {args.classes_file}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    model = build_model(
        img_size=args.img_size, channel_size=args.channel_size,
        patch_size=args.patch_size, channels=args.channels,
        blocks=args.blocks, layers_per_block=args.layers_per_block,
        nvp=args.nvp, num_classes=len(args.classes),
        device=device,
    )
    load_ckpt(model, args.ckpt, device=device)
    pi = load_prior(args.classes, args.prior_json)

    if args.cmd == "post":
        x = load_images(args.images, args.img_size, device)
        out = p_y_given_x(model, x, pi)   # [B,C]
        for i, pth in enumerate(args.images):
            probs = " ".join(f"{cls}:{out[i,j].item():.6f}" for j, cls in enumerate(args.classes))
            print(f"{pth}\t{probs}")

    elif args.cmd == "logpxy":
        x = load_images(args.images, args.img_size, device)
        y = torch.full((x.size(0),), int(args.y), device=device, dtype=torch.long)
        vals = log_p_x_given_y(model, x, y)  # [B]
        for pth, v in zip(args.images, vals.tolist()):
            print(f"{pth}\tlog_p(x|y={args.y})={v:.6f}")

    elif args.cmd == "logpx":
        x = load_images(args.images, args.img_size, device)
        vals = log_p_x(model, x, pi)  # [B]
        for pth, v in zip(args.images, vals.tolist()):
            print(f"{pth}\tlog_p(x)={v:.6f}")

if __name__ == "__main__":
    main()