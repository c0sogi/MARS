import os
import cv2
import math
import numpy as np
import pandas as pd
import torch
import albumentations as A
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_affine_transform, affine_transform


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


class KuzushijiDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True, debug_size=None):
        self.split = split
        self.input_size = Config.INPUT_SIZE
        self.output_size = self.input_size // 4
        self.num_classes = Config.NUM_CLASSES
        self.max_objs = Config.MAX_PREDS

        # Load ID mappings
        self.char2id, self.id2char = Config.get_class_mappings()

        # Determine paths
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
            self.img_dir = Config.INPUT_DIR
            self.is_train = True
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA_PATH
            self.img_dir = Config.INPUT_DIR
            self.is_train = False
        else:
            self.metadata_path = Config.TEST_METADATA_PATH
            self.img_dir = Config.INPUT_DIR
            self.is_train = False

        # Caching logic
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{split}_parsed.npy")

        self.data = []
        loaded = False

        if load_cached_data and os.path.exists(cache_path):
            try:
                self.data = np.load(cache_path, allow_pickle=True).tolist()
                loaded = True
            except Exception as e:
                print(f"Failed to load cache: {e}")

        if not loaded:
            df = pd.read_csv(self.metadata_path)

            # Process rows
            for _, row in df.iterrows():
                img_id = row["image_id"]
                file_path = row["file_path"]

                # Parse labels
                labels = []
                # Check if labels column exists and is not empty (Test set might have placeholder)
                if (
                    "labels" in row
                    and pd.notna(row["labels"])
                    and isinstance(row["labels"], str)
                ):
                    # For test set, we ignore the placeholder labels in sample_submission
                    if self.split == "test":
                        pass
                    else:
                        parts = row["labels"].strip().split(" ")
                        if len(parts) > 1:
                            # Format: Unicode X Y W H
                            for i in range(0, len(parts), 5):
                                try:
                                    u_char = parts[i]
                                    x = int(parts[i + 1])
                                    y = int(parts[i + 2])
                                    w = int(parts[i + 3])
                                    h = int(parts[i + 4])

                                    if u_char in self.char2id:
                                        cid = self.char2id[u_char]
                                        labels.append(
                                            {
                                                "class_id": cid,
                                                "bbox": [x, y, w, h],  # xywh
                                            }
                                        )
                                except (ValueError, IndexError):
                                    continue

                self.data.append(
                    {"image_id": img_id, "file_path": file_path, "labels": labels}
                )

            # Save to cache
            np.save(cache_path, np.array(self.data, dtype=object))

        if debug_size is not None:
            self.data = self.data[:debug_size]

        # Augmentations
        if self.is_train:
            self.transforms = A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ],
                bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
            )
        else:
            self.transforms = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ],
                bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item["file_path"])

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)

        h, w, c = img.shape

        # 1. Resize/Pad to Input Size using Affine Transform
        trans_input = get_affine_transform((h, w), self.input_size)

        # Warp image
        inp_img = cv2.warpAffine(
            img, trans_input, (self.input_size, self.input_size), flags=cv2.INTER_LINEAR
        )

        # Transform boxes
        boxes = []
        class_labels = []

        for label in item["labels"]:
            bx, by, bw, bh = label["bbox"]
            cid = label["class_id"]

            # Box corners
            x1, y1 = bx, by
            x2, y2 = bx + bw, by + bh

            # Transform corners
            p1 = affine_transform([x1, y1], trans_input)
            p2 = affine_transform([x2, y2], trans_input)

            # New box coordinates
            nx1 = min(p1[0], p2[0])
            ny1 = min(p1[1], p2[1])
            nx2 = max(p1[0], p2[0])
            ny2 = max(p1[1], p2[1])

            nw = nx2 - nx1
            nh = ny2 - ny1

            # Clip to image bounds
            nx1 = np.clip(nx1, 0, self.input_size - 1)
            ny1 = np.clip(ny1, 0, self.input_size - 1)
            nw = np.clip(nw, 0, self.input_size - 1 - nx1)
            nh = np.clip(nh, 0, self.input_size - 1 - ny1)

            if nw > 0 and nh > 0:
                boxes.append([nx1, ny1, nw, nh])
                class_labels.append(cid)

        # 2. Apply Albumentations (Augmentation + Normalization)
        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)

        augmented = self.transforms(
            image=inp_img, bboxes=boxes, class_labels=class_labels
        )
        inp_img = augmented["image"]
        boxes = augmented["bboxes"]
        class_labels = augmented["class_labels"]

        # To Tensor (C, H, W)
        inp_img = np.transpose(inp_img, (2, 0, 1)).astype(np.float32)

        # 3. Generate Targets (Heatmap, Regression)
        hm = np.zeros(
            (self.num_classes, self.output_size, self.output_size), dtype=np.float32
        )
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        num_objs = min(len(boxes), self.max_objs)

        for k in range(num_objs):
            bbox = boxes[k]
            cls_id = int(class_labels[k])

            # Box in input scale (1024)
            x, y, w, h_box = bbox

            # Project to output scale (256)
            # Center point
            ct = np.array([x + w / 2, y + h_box / 2], dtype=np.float32)
            ct_out = ct / 4.0

            ct_int = ct_out.astype(np.int32)

            # Clip center to output bounds
            ct_int[0] = np.clip(ct_int[0], 0, self.output_size - 1)
            ct_int[1] = np.clip(ct_int[1], 0, self.output_size - 1)

            # Gaussian Radius
            h_out = h_box / 4.0
            w_out = w / 4.0

            radius = gaussian_radius((math.ceil(h_out), math.ceil(w_out)))
            radius = max(0, int(radius))

            # Draw Heatmap
            draw_umich_gaussian(hm[cls_id], ct_int, radius)

            # Regression Targets
            wh[k] = 1.0 * w_out, 1.0 * h_out
            ind[k] = ct_int[1] * self.output_size + ct_int[0]
            reg[k] = ct_out - ct_int
            reg_mask[k] = 1

        ret = {
            "image": torch.from_numpy(inp_img),
            "hm": torch.from_numpy(hm),
            "wh": torch.from_numpy(wh),
            "reg": torch.from_numpy(reg),
            "ind": torch.from_numpy(ind),
            "reg_mask": torch.from_numpy(reg_mask),
            "image_id": item["image_id"],
        }

        return ret
