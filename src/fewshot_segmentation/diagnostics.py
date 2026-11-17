"""Same-image diagnostic utilities."""

import random
import numpy as np
import torch
import torch.nn.functional as F

from .config import device, IMAGE_SIZE
from .dataset import load_image, load_binary_mask
from .episodes import create_episode
from .models import extract_feature
from .prototypes import compute_kshot_prototypes, calculate_fg_bg_logits,calculate_prototypes
from .inference import self_support_refinement
from .losses import logits_to_prediction, calculate_metrics


def create_same_image_episode(class_id, class_images):
    samples = class_images[class_id]
    if len(samples) == 0:
        return None

    for _ in range(20):
        image_path, mask_path = random.choice(samples)
        image_np, tensor = load_image(image_path)
        mask = load_binary_mask(mask_path, class_id)

        if int(mask.sum()) > 0:
            return {
                "image": image_np,
                "tensor": tensor,
                "mask": mask,
                "class_id": class_id,
                "image_path": image_path,
            }

    return None


def same_image_diagnostic(model, class_id, class_images):
    episode = create_same_image_episode(class_id, class_images)
    if episode is None:
        print("Could not create same-image episode.")
        return None

    model.eval()

    with torch.no_grad():
        feature = extract_feature(model, episode["tensor"])

        mask_tensor = torch.from_numpy(episode["mask"]).float().to(device)

        fg_proto, bg_proto = calculate_prototypes(feature, mask_tensor)

        # Initial support-prototype matching
        initial_logits = calculate_fg_bg_logits(feature, fg_proto, bg_proto)

        # Rough probability: BEFORE SSP refinement
        rough_probability = torch.sigmoid(initial_logits)
        rough_probability = F.interpolate(
            rough_probability.unsqueeze(0).unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # SSP refinement
        final_logits = self_support_refinement(
            feature, fg_proto, bg_proto, initial_logits
        )

        # Final probability: AFTER SSP refinement
        final_probability = torch.sigmoid(final_logits)
        final_probability = F.interpolate(
            final_probability.unsqueeze(0).unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        final_prediction = (final_probability >= 0.5).astype(np.uint8)

        miou, fgiou, bgiou = calculate_metrics(final_prediction, episode["mask"])

    return {
        "image": episode["image"],
        "ground_truth": episode["mask"],
        "rough_probability": rough_probability,
        "final_probability": final_probability,
        "prediction": final_prediction,
        "mIoU": miou,
        "FGIoU": fgiou,
        "BGIoU": bgiou,
        "class_id": class_id,
        "image_path": episode["image_path"],
    }