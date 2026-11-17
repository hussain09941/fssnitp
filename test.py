"""
test.py — Kisi bhi class ke liye random image pe segmentation.

Usage:
    python test.py --class 8
    python test.py --class cat
    python test.py --class bus --episodes 3
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fewshot_segmentation.config import FOLDS, VOC_CLASSES, PROJECT_ROOT as PROJ_ROOT
from src.fewshot_segmentation.dataset import initialize_dataset
from src.fewshot_segmentation.models import create_backbone
from src.fewshot_segmentation.visualization import visualize_complete_episode


# ============================================================
# Helpers
# ============================================================
def parse_class_id(arg):
    """'cat' ya '8' dono accept karta hai."""
    try:
        cid = int(arg)
        if 1 <= cid <= 20:
            return cid
    except ValueError:
        pass

    name = arg.lower().strip()
    for cid, cname in enumerate(VOC_CLASSES):
        if cid > 0 and cname.lower() == name:
            return cid

    valid = ", ".join(f"{n}({i})" for i, n in enumerate(VOC_CLASSES) if i > 0)
    raise ValueError(f"Class '{arg}' nahi mila.\nValid: {valid}")


def get_fold_for_class(class_id):
    for fold_idx, classes in enumerate(FOLDS, 1):
        if class_id in classes:
            return fold_idx
    raise ValueError(f"Class {class_id} kisi fold mein nahi")


def load_model(fold_idx):
    ckpt = PROJ_ROOT / "checkpoints" / f"fold_{fold_idx}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Model nahi mila: {ckpt}\nPehle training chalao: python run.py"
        )
    model = create_backbone()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    print(f"✅ Loaded: {ckpt}")
    return model


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", "-c", dest="class_arg", required=True)
    parser.add_argument("--episodes", "-n", type=int, default=1)
    args = parser.parse_args()

    # Parse
    try:
        class_id = parse_class_id(args.class_arg)
    except ValueError as e:
        print(f"{e}")
        sys.exit(1)

    class_name = VOC_CLASSES[class_id]
    fold_idx = get_fold_for_class(class_id)

    print("=" * 60)
    print(f"Class : {class_name} (id={class_id})")
    print(f"Fold  : {fold_idx}  →  fold_{fold_idx}.pth")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)

    # Load
    class_images = initialize_dataset()
    model = load_model(fold_idx)

    # Run episodes — visualize_complete_episode internally does everything:
    #  - creates random episode (5 support + 1 query)
    #  - computes prototypes
    #  - SSP + ASBP refinement
    #  - plots 3 figures (support set, output, error map)
    #  - saves PNGs
    #  - returns dict with metrics
    for ep in range(1, args.episodes + 1):
        print(f"\n{'─' * 60}")
        print(f"Episode {ep}/{args.episodes}")
        print(f"{'─' * 60}")

        result = visualize_complete_episode(
            model=model,
            class_id=class_id,
            class_images=class_images,
            save_prefix=f"test_{class_name}_ep{ep}",
        )

        if result is None:
            print(f"❌ Episode {ep} failed")
            continue

        print(
            f"\n📊 mIoU {result['mIoU']*100:.2f}% | "
            f"FGIoU {result['FGIoU']*100:.2f}% | "
            f"BGIoU {result['BGIoU']*100:.2f}%"
        )

    print("\n✅ Done")


if __name__ == "__main__":
    main()