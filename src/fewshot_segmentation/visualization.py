"""Visualization of complete K-shot segmentation episodes."""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .config import PROJECT_ROOT, IMAGE_SIZE, VOC_CLASSES, device, K_SHOT
from .episodes import create_episode
from .inference import inference_episode, self_support_refinement
from .models import extract_feature
from .prototypes import compute_kshot_prototypes, calculate_fg_bg_logits
from .losses import logits_to_prediction, calculate_metrics


def visualize_complete_episode(
    model, class_id, class_images, k_shot=K_SHOT, save_prefix="fewshot"
):
    """
    Complete visualization:

        Support Image 1 ... Support Image K
        Support Mask  1 ... Support Mask  K

        Query Image | Ground Truth | Rough Prediction | Final Prediction

        Final Overlay | Error Map

    Rough Prediction = support prototype matching before SSP.
    Final Prediction = SSP-refined prediction.
    """
    # --------------------------------------------------------
    # Create a valid K-shot episode
    # --------------------------------------------------------
    episode = None
    for _ in range(50):
        episode = create_episode(class_id, class_images, k_shot)
        if episode is not None:
            break

    if episode is None:
        print("Could not create valid episode.")
        return None

    model.eval()

    with torch.no_grad():
        # ====================================================
        # STEP 1: K-shot support prototypes
        # ====================================================
        fg_prototype, bg_prototype = compute_kshot_prototypes(model, episode)
        if fg_prototype is None or bg_prototype is None:
            print("Prototype calculation failed.")
            return None

        # ====================================================
        # STEP 2: Query feature
        # ====================================================
        query_feature = extract_feature(model, episode["query_tensor"])

        # ====================================================
        # STEP 3: Initial FG/BG matching
        # ====================================================
        initial_logits = calculate_fg_bg_logits(
            query_feature, fg_prototype, bg_prototype
        )

        # ====================================================
        # STEP 4: ROUGH PREDICTION
        # ====================================================
        rough_probability = torch.sigmoid(initial_logits)
        rough_probability = F.interpolate(
            rough_probability.unsqueeze(0).unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        rough_prediction = (rough_probability >= 0.5).astype(np.uint8)

        # ====================================================
        # STEP 5: SSP REFINEMENT
        # ====================================================
        final_logits = self_support_refinement(
            query_feature, fg_prototype, bg_prototype, initial_logits
        )

        # ====================================================
        # STEP 6: FINAL PROBABILITY
        # ====================================================
        final_probability = torch.sigmoid(final_logits)
        final_probability = F.interpolate(
            final_probability.unsqueeze(0).unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # ====================================================
        # STEP 7: FINAL PREDICTION
        # ====================================================
        prediction = (final_probability >= 0.5).astype(np.uint8)

        # ====================================================
        # STEP 8: Ground truth
        # ====================================================
        query_gt = episode["query_mask"]

        # ====================================================
        # STEP 9: Metrics
        # ====================================================
        miou, fgiou, bgiou = calculate_metrics(prediction, query_gt)

    # ========================================================
    # PRINT RESULT
    # ========================================================
    print("\n" + "=" * 90)
    print("COMPLETE K-SHOT SEGMENTATION RESULT")
    print("=" * 90)
    print("Class:", VOC_CLASSES[class_id])
    print("K-shot:", len(episode["support_images"]))
    print(f"mIoU  : {miou * 100:.2f}%")
    print(f"FGIoU : {fgiou * 100:.2f}%")
    print(f"BGIoU : {bgiou * 100:.2f}%")
    print("=" * 90)

    # ========================================================
    # FINAL OVERLAY
    # ========================================================
    query_image = episode["query_image"].copy()
    segmented_image = query_image.copy()

    green = np.zeros_like(segmented_image)
    green[:, :, 1] = 255
    alpha = 0.5

    foreground = prediction == 1
    segmented_image[foreground] = (
        alpha * green[foreground] + (1 - alpha) * query_image[foreground]
    ).astype(np.uint8)

    # ========================================================
    # ERROR MAP
    # ========================================================
    gt_bool = query_gt == 1
    pred_bool = prediction == 1

    true_positive = gt_bool & pred_bool
    false_positive = ~gt_bool & pred_bool
    false_negative = gt_bool & ~pred_bool

    error_map = np.zeros_like(query_image)
    error_map[true_positive] = [0, 255, 0]
    error_map[false_positive] = [255, 0, 0]
    error_map[false_negative] = [0, 0, 255]

    # ========================================================
    # FIGURE 1: SUPPORT SET
    # ========================================================
    total_support = len(episode["support_images"])

    fig, axes = plt.subplots(
        2, total_support, figsize=(3.2 * total_support, 6.4)
    )

    if total_support == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for i in range(total_support):
        axes[0, i].imshow(episode["support_images"][i])
        axes[0, i].set_title(f"Support Image {i + 1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(
            episode["support_masks"][i], cmap="gray", vmin=0, vmax=1
        )
        axes[1, i].set_title(f"Support Mask {i + 1}")
        axes[1, i].axis("off")

    fig.suptitle(
        f"K-SHOT SUPPORT SET | Class = {VOC_CLASSES[class_id]}", fontsize=16
    )
    fig.tight_layout()

    support_filename = f"{save_prefix}_support_set.png"
    fig.savefig(f"results/{support_filename}", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # ========================================================
    # FIGURE 2: QUERY / GT / ROUGH / FINAL
    # ========================================================
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    axes[0].imshow(query_image)
    axes[0].set_title("Query Image", pad = 10)
    axes[0].axis("off")

    axes[1].imshow(query_gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth", pad = 10)
    axes[1].axis("off")

    axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Final Prediction\nFGIoU = {fgiou * 100:.2f}%", pad =10)
    axes[2].axis("off")


    fig.suptitle(
        f"SSP OUTPUT | Class = {VOC_CLASSES[class_id]} | "
        f"mIoU = {miou * 100:.2f}%",
        fontsize=16,
    )
    fig.tight_layout()

    prediction_filename = f"{save_prefix}_ssp_output.png"
    fig.savefig(f"results/{prediction_filename}", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # ========================================================
    # FIGURE 3: OVERLAY + ERROR MAP
    # ========================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    axes[0].imshow(segmented_image)
    axes[0].set_title(f"Final Segmentation Overlay\nmIoU = {miou * 100:.2f}%")
    axes[0].axis("off")

    axes[1].imshow(error_map)
    axes[1].set_title("Error Map\nGreen = TP | Red = FP | Blue = FN")
    axes[1].axis("off")

    fig.suptitle(f"DIAGNOSTICS | {VOC_CLASSES[class_id]}", fontsize=16)
    fig.tight_layout()

    error_filename = f"{save_prefix}_overlay_error.png"
    fig.savefig(f"results/{error_filename}", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("\n" + "=" * 70)
    print("VISUALIZATION FILES SAVED")
    print("=" * 70)
    print("1.", support_filename)
    print("2.", prediction_filename)
    print("3.", error_filename)
    print("=" * 70)

    return {
        "episode": episode,
        "rough_probability": rough_probability,
        "rough_prediction": rough_prediction,
        "final_probability": final_probability,
        "prediction": prediction,
        "ground_truth": query_gt,
        "mIoU": miou,
        "FGIoU": fgiou,
        "BGIoU": bgiou,
    }