"""Inference — SSP + ASBP + iterative refinement (paper Table 3 'SSP refine')."""
import numpy as np
import torch
import torch.nn.functional as F

from .config import IMAGE_SIZE
from .episodes import create_episode
from .models import extract_feature
from .prototypes import (
    compute_kshot_prototypes, calculate_fg_bg_logits,
    compute_self_support_fg_prototype,
    compute_adaptive_self_support_bg_prototype,
    blend_prototypes, match_with_blended,
)
from .losses import logits_to_prediction, calculate_metrics


def self_support_refinement(query_feature, support_fg, support_bg, initial_logits):
    q_fg = compute_self_support_fg_prototype(query_feature, initial_logits)
    q_bg = compute_adaptive_self_support_bg_prototype(query_feature, initial_logits)

    if q_fg is None or q_bg is None:
        return initial_logits

    blended_fg = blend_prototypes(support_fg, q_fg)
    blended_bg = blend_prototypes(support_bg, q_bg)
    return match_with_blended(query_feature, blended_fg, blended_bg)


def iterative_refine(query_feature, support_fg, support_bg, logits, n_iters=1):
    for _ in range(n_iters):
        logits = self_support_refinement(
            query_feature, support_fg, support_bg, logits
        )
    return logits


def inference_episode(model, class_id, class_images, refine_iters=1):
    episode = create_episode(class_id, class_images)
    if episode is None:
        return None

    model.eval()
    with torch.no_grad():
        fg_prototype, bg_prototype = compute_kshot_prototypes(model, episode)
        if fg_prototype is None:
            return None

        query_feature = extract_feature(model, episode["query_tensor"])

        initial_logits = calculate_fg_bg_logits(
            query_feature, fg_prototype, bg_prototype
        )

        logits = iterative_refine(
            query_feature, fg_prototype, bg_prototype,
            initial_logits, n_iters=refine_iters,
        )

        prediction = logits_to_prediction(
            logits, output_size=(IMAGE_SIZE, IMAGE_SIZE)
        )
        miou, fgiou, bgiou = calculate_metrics(prediction, episode["query_mask"])

        if not (np.isfinite(miou) and np.isfinite(fgiou)):
            return None

        probability = torch.sigmoid(logits)
        probability = F.interpolate(
            probability.unsqueeze(0).unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False,
        ).squeeze().cpu().numpy()

    return {
        "mIoU": miou, "FGIoU": fgiou, "BGIoU": bgiou,
        "support_image": episode["support_image"],
        "support_mask": episode["support_mask"],
        "support_images": episode["support_images"],
        "support_masks": episode["support_masks"],
        "query_image": episode["query_image"],
        "query_mask": episode["query_mask"],
        "prediction": prediction,
        "probability": probability,
        "class_id": class_id,
    }