"""Few-shot episode sampling (paper-aligned)."""
import os
import random
from .config import K_SHOT, MIN_FG_PIXELS
from .dataset import load_image, load_binary_mask


def create_episode(class_id, class_images, k_shot=K_SHOT):
    if class_id not in class_images:
        return None

    samples = class_images[class_id]
    if len(samples) < k_shot + 1:
        return None

    reliable = []
    for img_path, mask_path in samples:
        if not (os.path.exists(img_path) and os.path.exists(mask_path)):
            continue
        mask = load_binary_mask(mask_path, class_id)
        if int(mask.sum()) >= MIN_FG_PIXELS:
            reliable.append((img_path, mask_path))

    if len(reliable) < k_shot + 1:
        return None

    chosen = random.sample(reliable, k_shot + 1)
    support_items = chosen[:k_shot]
    query_item = chosen[k_shot]

    support_images, support_tensors, support_masks = [], [], []
    for sp_img, sp_msk in support_items:
        img, ten = load_image(sp_img)
        m = load_binary_mask(sp_msk, class_id)
        if int(m.sum()) < MIN_FG_PIXELS:
            return None
        support_images.append(img)
        support_tensors.append(ten)
        support_masks.append(m)

    if len(support_tensors) != k_shot:
        return None

    q_img, q_msk = query_item
    query_image, query_tensor = load_image(q_img)
    query_mask = load_binary_mask(q_msk, class_id)
    if int(query_mask.sum()) < MIN_FG_PIXELS:
        return None

    return {
        "support_images": support_images,
        "support_tensors": support_tensors,
        "support_masks": support_masks,
        "support_image": support_images[0],
        "support_mask": support_masks[0],
        "query_image": query_image,
        "query_tensor": query_tensor,
        "query_mask": query_mask,
        "class_id": class_id,
    }