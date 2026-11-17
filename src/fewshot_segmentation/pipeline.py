"""End-to-end 4-fold training and unseen-class evaluation (SSP paper)."""
import os
import random
import numpy as np
import torch

from .config import (
    PROJECT_ROOT, device, TRAIN_ITERATIONS, BATCH_PAIRS,
    LEARNING_RATE, MOMENTUM, WEIGHT_DECAY,
    LR_DECAY_STEP, LR_DECAY_GAMMA,
    EPISODES_PER_TEST_FOLD, VOC_CLASSES, FOLDS,
)
from .dataset import initialize_dataset
from .models import create_backbone
from .training import train_episode
from .inference import inference_episode
from .evaluation import print_model_info
from .diagnostics import same_image_diagnostic
from .visualization import visualize_complete_episode


def run_pipeline(run_diagnostics=True, run_visualization=True):
    print("Device:", device)
    class_images = initialize_dataset()

    def train_one(model, optimizer, class_id):
        return train_episode(model, optimizer, class_id, class_images)

    def infer_one(model, class_id, refine_iters=1):
        return inference_episode(model, class_id, class_images,
                                 refine_iters=refine_iters)

    all_trained_models = {}
    training_results = {}

    for fold_idx, test_classes in enumerate(FOLDS, 1):
        train_classes = [c for c in range(1, 21) if c not in test_classes]
        print(f"\nFOLD {fold_idx} | test: {[VOC_CLASSES[c] for c in test_classes]}")

        model = create_backbone()
        print_model_info(model)

        # Paper: SGD, momentum 0.9, lr 1e-3, decay 10x every 2000 iter
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA,
        )

        losses, mious = [], []

        for it in range(1, TRAIN_ITERATIONS + 1):
            batch_loss, batch_miou, valid = 0.0, 0.0, 0

            for _ in range(BATCH_PAIRS):
                class_id = random.choice(train_classes)
                result = train_one(model, optimizer, class_id)
                if result is None:
                    continue
                batch_loss += result["loss"]
                batch_miou += result["mIoU"]
                valid += 1

            scheduler.step()

            if valid == 0:
                continue

            avg_loss = batch_loss / valid
            avg_miou = batch_miou / valid
            losses.append(avg_loss)
            mious.append(avg_miou)

            if it == 1 or it % 50 == 0:
                print(
                    f"Fold {fold_idx} | iter {it:04d}/{TRAIN_ITERATIONS} | "
                    f"loss {avg_loss:.4f} | mIoU {avg_miou * 100:.2f}%"
                )

        mean_loss = np.mean(losses) if losses else np.nan
        mean_miou = np.mean(mious) if mious else np.nan
        training_results[fold_idx] = {"loss": mean_loss, "mIoU": mean_miou}
        all_trained_models[fold_idx] = model
        print(f"Fold {fold_idx} training | loss {mean_loss:.4f} | "
              f"mIoU {mean_miou * 100:.2f}%")

    # Save
    MODEL_DIR = str(PROJECT_ROOT / "checkpoints")
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fold_idx, model in all_trained_models.items():
        path = os.path.join(MODEL_DIR, f"fold_{fold_idx}.pth")
        torch.save(model.state_dict(), path)
        print("Saved:", path)

    # Inference
    inference_results = {}
    for fold_idx, test_classes in enumerate(FOLDS, 1):
        model = all_trained_models[fold_idx]
        mious, fgious, bgious = [], [], []
        for ep in range(1, EPISODES_PER_TEST_FOLD + 1):
            cid = random.choice(test_classes)
            r = infer_one(model, cid, refine_iters=1)
            if r is None:
                continue
            mious.append(r["mIoU"])
            fgious.append(r["FGIoU"])
            bgious.append(r["BGIoU"])

        inference_results[fold_idx] = {
            "mIoU": np.mean(mious) if mious else np.nan,
            "FGIoU": np.mean(fgious) if fgious else np.nan,
            "BGIoU": np.mean(bgious) if bgious else np.nan,
        }
        print(f"Fold {fold_idx} test | mIoU {inference_results[fold_idx]['mIoU']*100:.2f}%")

    final_miou = np.nanmean([inference_results[f]["mIoU"] for f in inference_results])
    print(f"\nFINAL TEST mIoU: {final_miou * 100:.2f}%")

    if run_diagnostics:
        same_image_diagnostic(all_trained_models[1], 1, class_images)
    if run_visualization:
        visualize_complete_episode(
            all_trained_models[1], random.choice(FOLDS[0]), class_images
        )

    return {
        "models": all_trained_models,
        "training_results": training_results,
        "inference_results": inference_results,
        "class_images": class_images,
    }


if __name__ == "__main__":
    run_pipeline()