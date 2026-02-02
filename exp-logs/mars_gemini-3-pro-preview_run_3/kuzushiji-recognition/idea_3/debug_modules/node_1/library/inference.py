import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A

from library.config import Config
from library.utils import ctdet_decode, nms
from library.models import KuzushijiDetector, KuzushijiClassifier


class TiledDetector:
    def __init__(self, weights_path, device):
        self.device = device
        self.model = KuzushijiDetector(pretrained=False)

        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Detector weights loaded from {weights_path}")
        else:
            print(
                f"Warning: Detector weights not found at {weights_path}. Using random init."
            )

        self.model.to(device)
        self.model.eval()

        self.transform = A.Compose(
            [A.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD), ToTensorV2()]
        )

    def preprocess_tile(self, tile):
        # Pad if smaller than input size
        h, w, _ = tile.shape
        pad_h = max(0, Config.DETECTOR_INPUT_SIZE[0] - h)
        pad_w = max(0, Config.DETECTOR_INPUT_SIZE[1] - w)

        if pad_h > 0 or pad_w > 0:
            tile = cv2.copyMakeBorder(
                tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            )

        transformed = self.transform(image=tile)
        return transformed["image"].unsqueeze(0).to(self.device)

    def detect(self, image):
        H, W, _ = image.shape
        tile_size = Config.TILE_SIZE
        overlap = Config.TILE_OVERLAP
        stride = int(tile_size * (1 - overlap))

        # Calculate tile coordinates
        x_steps = []
        x = 0
        while x < W:
            x_steps.append(min(x, W - tile_size) if W > tile_size else 0)
            x += stride
            if x_steps[-1] == max(0, W - tile_size):
                break

        y_steps = []
        y = 0
        while y < H:
            y_steps.append(min(y, H - tile_size) if H > tile_size else 0)
            y += stride
            if y_steps[-1] == max(0, H - tile_size):
                break

        # Remove duplicates if image is small
        x_steps = sorted(list(set(x_steps)))
        y_steps = sorted(list(set(y_steps)))

        all_detections = []

        with torch.no_grad():
            for y_off in y_steps:
                for x_off in x_steps:
                    # Extract tile
                    tile = image[y_off : y_off + tile_size, x_off : x_off + tile_size]

                    # Preprocess
                    input_tensor = self.preprocess_tile(tile)

                    # Forward
                    hm, wh, reg = self.model(input_tensor)

                    # Decode
                    # Output is (1, K, 6) -> [x1, y1, x2, y2, score, class]
                    # Coordinates are in feature scale
                    dets = ctdet_decode(hm, wh, reg, K=100)  # Top 100 per tile
                    dets = dets.cpu().numpy().reshape(-1, 6)

                    # Filter by score
                    mask = dets[:, 4] >= Config.SCORE_THRESHOLD
                    dets = dets[mask]

                    if len(dets) > 0:
                        # Transform coordinates
                        # 1. Scale up from feature map to tile input size
                        dets[:, 0:4] *= Config.DETECTOR_OUTPUT_STRIDE

                        # 2. Add global offset
                        dets[:, 0] += x_off
                        dets[:, 2] += x_off
                        dets[:, 1] += y_off
                        dets[:, 3] += y_off

                        # 3. Clip to image boundaries
                        dets[:, 0] = np.clip(dets[:, 0], 0, W)
                        dets[:, 2] = np.clip(dets[:, 2], 0, W)
                        dets[:, 1] = np.clip(dets[:, 1], 0, H)
                        dets[:, 3] = np.clip(dets[:, 3], 0, H)

                        all_detections.append(dets)

        if not all_detections:
            return np.array([])

        all_detections = np.concatenate(all_detections, axis=0)

        # Global NMS
        # dets format: x1, y1, x2, y2, score, class
        keep_indices = nms(
            all_detections[:, 0:4],
            all_detections[:, 4],
            overlap_thresh=Config.NMS_IOU_THRESHOLD,
        )
        final_detections = all_detections[keep_indices]

        # Limit total detections per page
        if len(final_detections) > Config.MAX_DETECTIONS_PER_PAGE:
            # Sort by score descending
            indices = np.argsort(final_detections[:, 4])[::-1]
            final_detections = final_detections[
                indices[: Config.MAX_DETECTIONS_PER_PAGE]
            ]

        return final_detections


class BatchClassifier:
    def __init__(self, weights_path, class_map_path, device):
        self.device = device

        # Load Class Map
        if os.path.exists(class_map_path):
            self.class_map = np.load(class_map_path, allow_pickle=True).item()
            self.id_to_code = {v: k for k, v in self.class_map.items()}
            num_classes = len(self.class_map)
        else:
            print(f"Warning: Class map not found at {class_map_path}. Using dummy map.")
            self.class_map = {}
            self.id_to_code = {}
            num_classes = 4000  # Fallback

        self.model = KuzushijiClassifier(num_classes=num_classes, pretrained=False)

        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Classifier weights loaded from {weights_path}")
        else:
            print(f"Warning: Classifier weights not found at {weights_path}.")

        self.model.to(device)
        self.model.eval()

        self.transform = A.Compose(
            [A.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD), ToTensorV2()]
        )

    def classify(self, image, bboxes):
        if len(bboxes) == 0:
            return []

        crops = []
        valid_indices = []

        H, W, _ = image.shape

        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = box[0:4].astype(int)

            # Ensure valid crop
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            crop = cv2.resize(crop, Config.CLASSIFIER_INPUT_SIZE)

            transformed = self.transform(image=crop)
            crops.append(transformed["image"])
            valid_indices.append(i)

        if not crops:
            return []

        # Batch processing
        batch_size = Config.CLASSIFIER_BATCH_SIZE
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(crops), batch_size):
                batch_crops = torch.stack(crops[i : i + batch_size]).to(self.device)
                outputs = self.model(batch_crops)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())

        results = []
        for idx, pred_id in zip(valid_indices, all_preds):
            code = self.id_to_code.get(pred_id, "U+0000")  # Fallback
            box = bboxes[idx]
            results.append((code, box))

        return results


def generate_submission(
    test_metadata_path=Config.TEST_METADATA,
    detector_weights=os.path.join(Config.WORKING_DIR, "detector_best.pth"),
    classifier_weights=os.path.join(Config.WORKING_DIR, "classifier_best.pth"),
    output_csv="./submission/submission.csv",
):
    print("Starting submission generation...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Load Metadata
    if not os.path.exists(test_metadata_path):
        print(f"Error: Metadata not found at {test_metadata_path}")
        return

    df_test = pd.read_csv(test_metadata_path)

    # Initialize Models
    device = Config.DEVICE
    detector = TiledDetector(detector_weights, device)
    classifier = BatchClassifier(classifier_weights, Config.CLASS_MAP_PATH, device)

    submission_rows = []

    for idx, row in df_test.iterrows():
        image_id = row["image_id"]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        if not os.path.exists(file_path):
            submission_rows.append({"image_id": image_id, "labels": ""})
            continue

        # Load Image
        image = cv2.imread(file_path)
        if image is None:
            submission_rows.append({"image_id": image_id, "labels": ""})
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Detect
        detections = detector.detect(image)

        # 2. Classify
        labeled_preds = classifier.classify(image, detections)

        # 3. Format Output
        label_strs = []
        for code, box in labeled_preds:
            x1, y1, x2, y2 = box[0:4]

            # Calculate Center
            cx = int(x1 + (x2 - x1) / 2)
            cy = int(y1 + (y2 - y1) / 2)

            label_strs.append(f"{code} {cx} {cy}")

        label_string = " ".join(label_strs)
        submission_rows.append({"image_id": image_id, "labels": label_string})

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df_test)} images.")

    # Save Submission
    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")
