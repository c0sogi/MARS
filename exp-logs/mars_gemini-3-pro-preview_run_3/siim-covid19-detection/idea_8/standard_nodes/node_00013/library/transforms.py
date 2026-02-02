import cv2
import numpy as np
import torch
import random
from library.config import Config
from library.utils import box_xyxy_to_cxcywh


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class LetterboxResize:
    """
    Resizes an image to a target size while preserving aspect ratio.
    Pads the shorter dimension with zeros (or a specific color) to create a square image.
    Adjusts bounding boxes accordingly.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        h, w = image.shape[:2]

        # Calculate scale factor to fit the longest dimension into size
        scale = self.size / max(h, w)

        # Resize image
        if scale != 1:
            interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            image = cv2.resize(image, (new_w, new_h), interpolation=interp)
        else:
            new_w, new_h = w, h

        # Calculate padding
        dw = self.size - new_w
        dh = self.size - new_h

        # Divide padding for centering
        top = dh // 2
        bottom = dh - top
        left = dw // 2
        right = dw - left

        # Add border (padding)
        # Using 0 (black) for padding
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        # Adjust bounding boxes if they exist
        if target is not None and "boxes" in target:
            boxes = target["boxes"]
            if len(boxes) > 0:
                # boxes are expected to be numpy array [N, 4] in x1, y1, x2, y2 format
                # Scale
                boxes[:, [0, 2]] *= scale
                boxes[:, [1, 3]] *= scale

                # Shift by padding
                boxes[:, [0, 2]] += left
                boxes[:, [1, 3]] += top

                target["boxes"] = boxes

        return image, target


class RandomHorizontalFlip:
    """
    Horizontally flips the given image randomly with a given probability.
    Adjusts bounding boxes accordingly.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            h, w = image.shape[:2]

            # Flip image
            image = cv2.flip(image, 1)

            # Flip boxes
            if target is not None and "boxes" in target:
                boxes = target["boxes"]
                if len(boxes) > 0:
                    # x1_new = w - x2_old
                    # x2_new = w - x1_old
                    # We need to copy to avoid issues with in-place modification during swap
                    boxes_copy = boxes.copy()
                    boxes[:, 0] = w - boxes_copy[:, 2]
                    boxes[:, 2] = w - boxes_copy[:, 0]
                    target["boxes"] = boxes

        return image, target


class ToTensor:
    """
    Converts numpy image to torch tensor (C, H, W) in range [0, 1].
    Converts target boxes to tensor, normalizes them to [0, 1], and converts format to (cx, cy, w, h).
    """

    def __call__(self, image, target):
        # Image: HWC -> CHW, Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        if target is not None:
            # Handle Boxes
            if "boxes" in target:
                boxes = target["boxes"]
                if len(boxes) > 0:
                    boxes = torch.from_numpy(boxes).float()

                    # Get image dimensions (tensor shape is C, H, W)
                    h, w = image.shape[-2:]

                    # Convert xyxy to cxcywh using library utility
                    boxes = box_xyxy_to_cxcywh(boxes)

                    # Normalize coordinates to [0, 1] relative to image size
                    boxes[:, 0] /= w  # cx
                    boxes[:, 1] /= h  # cy
                    boxes[:, 2] /= w  # w
                    boxes[:, 3] /= h  # h

                    # Clamp to ensure numerical stability within [0, 1]
                    boxes = torch.clamp(boxes, min=0.0, max=1.0)

                    target["boxes"] = boxes
                else:
                    target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

            # Handle Labels
            if "labels" in target:
                target["labels"] = torch.from_numpy(target["labels"]).long()

            # Handle Study Label
            if "study_label" in target:
                if isinstance(target["study_label"], (int, np.integer)):
                    target["study_label"] = torch.tensor(
                        target["study_label"], dtype=torch.long
                    )
                elif isinstance(target["study_label"], np.ndarray):
                    target["study_label"] = torch.from_numpy(
                        target["study_label"]
                    ).long()

        return image, target


class Normalize:
    """
    Normalizes a tensor image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, image, target):
        # image is already a tensor (C, H, W)
        image = (image - self.mean) / self.std
        return image, target


def get_transforms(split):
    """
    Returns the transformation pipeline for a given data split.

    Args:
        split (str): 'train', 'val', or 'test'.
    """
    # Standard ImageNet normalization stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transforms = []

    # 1. Geometric Transformations (Resize & Flip)
    transforms.append(LetterboxResize(Config.IMG_SIZE))

    if split == "train":
        transforms.append(RandomHorizontalFlip(p=0.5))

    # 2. Conversion to Tensor & Box Formatting (xyxy -> cxcywh normalized)
    transforms.append(ToTensor())

    # 3. Pixel Normalization
    transforms.append(Normalize(mean, std))

    return Compose(transforms)
