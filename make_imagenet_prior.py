#!/usr/bin/env python3
import argparse, json, os, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def read_classes(path: str) -> List[str]:
    classes = [ln.strip().split()[0] for ln in Path(path).read_text().splitlines() if ln.strip()]
    if len(classes) == 0:
        raise ValueError(f"No classes found in {path}")
    # enforce uniqueness & order
    if len(set(classes)) != len(classes):
        raise ValueError("classes_file has duplicates; please deduplicate.")
    return classes

def count_dir_images(root: Path, class_id: str, recursive: bool) -> int:
    """Count images for class_id under root/class_id[/...] with known extensions."""
    cls_dir = root / class_id
    if not cls_dir.exists():
        return 0
    n = 0
    if recursive:
        for dp, _, files in os.walk(cls_dir):
            for f in files:
                if Path(f).suffix.lower() in IMAGE_EXTS:
                    n += 1
    else:
        for f in cls_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                n += 1
    return n

def count_from_index(index_file: Path, classes_set: set) -> Dict[str, int]:
    """
    Read lines of the form: "<path> <label>"
    Counts only lines whose label is in classes_set.
    """
    counts: Dict[str, int] = {c: 0 for c in classes_set}
    with index_file.open("r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # allow spaces in path by splitting from the right once
            # e.g., ".../my image.jpg n01440764"
            try:
                path_part, label = line.rsplit(maxsplit=1)
            except ValueError:
                raise ValueError(f"Malformed line {i} in {index_file}: '{line}'")
            if label in counts:
                counts[label] += 1
    return counts

def main():
    ap = argparse.ArgumentParser("Compute ImageNet class priors JSON")
    ap.add_argument("--classes_file", required=True,
                    help="Text file with one class ID per line in the exact model order (e.g., WNIDs).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--train_root",
                   help="Root of ImageNet train folder with class subdirs (e.g., /path/ILSVRC/train).")
    g.add_argument("--index_file",
                   help="Text file with lines '<image_path> <class_id>' for all training images.")
    ap.add_argument("--recursive", action="store_true", default=True,
                    help="When using --train_root, recurse into subfolders (default: True).")
    ap.add_argument("--out", required=True,
                    help="Output JSON path (e.g., data/imagenet_prior.json).")
    ap.add_argument("--strict", action="store_true", default=False,
                    help="If set, error on any class with zero images or missing directory.")
    args = ap.parse_args()

    classes = read_classes(args.classes_file)
    C = len(classes)
    classes_set = set(classes)

    # Count images per class
    counts: Dict[str, int] = {c: 0 for c in classes}

    if args.train_root:
        root = Path(args.train_root)
        if not root.exists():
            print(f"[error] train_root not found: {root}", file=sys.stderr)
            sys.exit(1)
        for c in classes:
            n = count_dir_images(root, c, recursive=args.recursive)
            counts[c] = n
    else:
        index_file = Path(args.index_file)
        if not index_file.exists():
            print(f"[error] index_file not found: {index_file}", file=sys.stderr)
            sys.exit(1)
        idx_counts = count_from_index(index_file, classes_set)
        # Keep the explicit order from classes_file
        for c in classes:
            counts[c] = idx_counts.get(c, 0)

    total = sum(counts.values())

    # Sanity checks / messages
    missing = [c for c, n in counts.items() if n == 0]
    print(f"[info] classes: {C}")
    print(f"[info] total images counted: {total}")
    if missing:
        print(f"[warn] {len(missing)} classes have zero images: (showing up to first 10)\n       {missing[:10]}")
        if args.strict:
            print("[error] --strict enabled and some classes have zero images.", file=sys.stderr)
            sys.exit(2)

    # Build prior
    if total == 0:
        print("[warn] No images counted; falling back to uniform prior.")
        pi = {c: 1.0 / C for c in classes}
    else:
        pi = {c: counts[c] / total for c in classes}

    # Emit JSON
    out_obj = {
        "order": classes,
        "pi": pi,
        "meta": {
            "source": "train_root" if args.train_root else "index_file",
            "total_images": int(total),
            "zero_count_classes": len(missing),
        }
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2))
    print(f"[done] wrote prior JSON to {out_path}")

if __name__ == "__main__":
    main()