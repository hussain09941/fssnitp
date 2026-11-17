"""Pascal VOC dataset discovery, decoding and preprocessing."""

import os
import numpy as np
import cv2
from PIL import Image
from tqdm.auto import tqdm
from torchvision import models

from .config import PROJECT_ROOT, IMAGE_SIZE, VOC_CLASSES, device


def find_voc_root(base_path):
    possible_paths = [
        os.path.join(base_path, "VOC2012_train_val", "VOC2012_train_val"),
        os.path.join(base_path, "VOC2012_train_val"),
        base_path,
    ]

    for path in possible_paths:
        jpeg = os.path.join(path, "JPEGImages")
        masks = os.path.join(path, "SegmentationClass")
        if os.path.isdir(jpeg) and os.path.isdir(masks):
            return path

    # Recursive search
    for root, dirs, files in os.walk(base_path):
        if (
            os.path.basename(root) == "JPEGImages"
            and os.path.isdir(os.path.join(os.path.dirname(root), "SegmentationClass"))
        ):
            return os.path.dirname(root)

    raise FileNotFoundError(
        "Could not find VOC2012 JPEGImages/SegmentationClass folders."
    )


def decode_voc_mask(mask_path):
    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    # Standard VOC indexed PNG
    if mask_array.ndim == 2:
        return mask_array.astype(np.uint8)

    # RGB fallback
    rgb = np.array(mask.convert("RGB"))
    class_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

    # VOC color decoding
    for class_id in range(21):
        color = [0, 0, 0]
        value = class_id
        for j in range(8):
            color[0] |= ((value >> 0) & 1) << j
            color[1] |= ((value >> 1) & 1) << j
            color[2] |= ((value >> 2) & 1) << j
            value >>= 3

        matches = np.all(rgb == np.array(color), axis=2)
        class_mask[matches] = class_id

    return class_mask


def initialize_dataset():
    """Locate local VOC2012 dataset and build class-wise image/mask pairs."""
    default_path = PROJECT_ROOT / "data"
    local_path = os.environ.get("VOC_DATASET_PATH")
    dataset_path = local_path if local_path else str(default_path)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    voc_root = find_voc_root(dataset_path)

    jpeg_dir = os.path.join(voc_root, "JPEGImages")
    mask_dir = os.path.join(voc_root, "SegmentationClass")

    image_files = sorted(
        f for f in os.listdir(jpeg_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    mask_files = sorted(
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(".png")
    )

    image_dict = {
        os.path.splitext(f)[0]: os.path.join(jpeg_dir, f)
        for f in image_files
    }
    mask_dict = {
        os.path.splitext(f)[0]: os.path.join(mask_dir, f)
        for f in mask_files
    }

    common_ids = sorted(set(image_dict) & set(mask_dict))
    image_mask_pairs = [(image_dict[i], mask_dict[i]) for i in common_ids]

    class_images = {class_id: [] for class_id in range(1, 21)}

    print(f"Dataset path: {dataset_path}")
    print(f"VOC root: {voc_root}")
    print(f"Images: {len(image_files)} | Masks: {len(mask_files)}")
    print(f"Matching image-mask pairs: {len(image_mask_pairs)}")
    print("Building class-wise image lists...")

    for image_path, mask_path in tqdm(image_mask_pairs):
        mask = decode_voc_mask(mask_path)
        for class_id in np.unique(mask):
            class_id = int(class_id)
            if 1 <= class_id <= 20:
                class_images[class_id].append((image_path, mask_path))

    print("\nClass-wise image counts:")
    for class_id in range(1, 21):
        print(
            f"{class_id:2d} | {VOC_CLASSES[class_id]:15s} | "
            f"{len(class_images[class_id]):5d}"
        )

    return class_images


def load_image(image_path):
    weights = models.ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    image_np = np.array(resized_image)
    tensor = preprocess(resized_image)
    return image_np, tensor


def load_binary_mask(mask_path, class_id):
    mask = decode_voc_mask(mask_path)
    binary_mask = (mask == class_id).astype(np.uint8)
    return cv2.resize(
        binary_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST
    )