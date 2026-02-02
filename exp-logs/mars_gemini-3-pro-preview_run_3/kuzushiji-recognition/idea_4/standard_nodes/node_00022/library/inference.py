import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.ops import nms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.models import CenterNetDetector, ResNetClassifier
from library.utils import set_seed


def decode_centernet(hm, wh, reg, stride=4, score_thresh=0.3, K=1200):
    """
    Decodes CenterNet output tensors into bounding boxes.
    """
    batch, _, height, width = hm.shape

    # 1. Max Pooling to find peaks (3x3)
    hm_max = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    keep = (hm_max == hm).float()
    hm = hm * keep

    # 2. Top K
    hm = hm.view(batch, -1)
    scores, inds = torch.topk(hm, K)

    # Convert indices to x, y
    ys = inds.div(width, rounding_mode="floor").float()
    xs = (inds % width).float()

    # 3. Get reg and wh
    reg = reg.permute(0, 2, 3, 1).contiguous().view(batch, -1, 2)
    wh = wh.permute(0, 2, 3, 1).contiguous().view(batch, -1, 2)

    # Gather based on inds
    reg_x = torch.gather(reg[..., 0], 1, inds)
    reg_y = torch.gather(reg[..., 1], 1, inds)

    w = torch.gather(wh[..., 0], 1, inds)
    h = torch.gather(wh[..., 1], 1, inds)

    # 4. Refine coordinates
    xs = (xs + reg_x) * stride
    ys = (ys + reg_y) * stride

    # 5. Filter by score
    mask = scores > score_thresh

    detections = []
    for b in range(batch):
        b_mask = mask[b]
        if b_mask.sum() == 0:
            detections.append(None)
            continue

        b_scores = scores[b][b_mask]
        b_xs = xs[b][b_mask]
        b_ys = ys[b][b_mask]
        b_w = w[b][b_mask]
        b_h = h[b][b_mask]

        # Create boxes: x1, y1, x2, y2 (for NMS)
        x1 = b_xs - b_w / 2
        y1 = b_ys - b_h / 2
        x2 = b_xs + b_w / 2
        y2 = b_ys + b_h / 2

        # Stack: x1, y1, x2, y2, score
        dets = torch.stack([x1, y1, x2, y2, b_scores], dim=1)
        detections.append(dets)

    return detections


class ResizeDetector:
    """
    Detects objects by resizing the full image to the detector input size,
    then mapping coordinates back to original resolution.
    Cite solution_lesson_node_00021: Decoupling Resolution Strategies.
    """

    def __init__(self, model, device, input_size=1024, conf_thresh=0.3):
        self.model = model
        self.device = device
        self.input_size = input_size
        self.conf_thresh = conf_thresh

        self.transform = A.Compose(
            [
                A.Resize(height=input_size, width=input_size),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )

    def detect(self, image):
        """
        Runs detection on a single full-resolution image.
        Returns tensor of shape (N, 5) -> [x1, y1, x2, y2, score] in global coords.
        """
        orig_h, orig_w, _ = image.shape

        # Transform (Resize)
        aug = self.transform(image=image)
        input_tensor = aug["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            hm, wh, reg = self.model(input_tensor)

        # Decode in 1024x1024 space
        # Output stride is 4 relative to input 1024 (feature map 256)
        detections = decode_centernet(
            hm,
            wh,
            reg,
            stride=Config.DETECTOR_STRIDE,
            score_thresh=self.conf_thresh,
        )

        # Unpack batch (size 1)
        dets = detections[0]

        if dets is None or dets.size(0) == 0:
            return torch.empty((0, 5), device=self.device)

        # Map back to original resolution
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        dets[:, 0] *= scale_x
        dets[:, 1] *= scale_y
        dets[:, 2] *= scale_x
        dets[:, 3] *= scale_y

        # Clip to original image size
        dets[:, 0] = torch.clamp(dets[:, 0], 0, orig_w)
        dets[:, 1] = torch.clamp(dets[:, 1], 0, orig_h)
        dets[:, 2] = torch.clamp(dets[:, 2], 0, orig_w)
        dets[:, 3] = torch.clamp(dets[:, 3], 0, orig_h)

        return dets


class InferencePipeline:
    def __init__(self):
        set_seed(Config.SEED)
        self.device = Config.DEVICE

        # 1. Load Class Map
        if os.path.exists(Config.CLASS_MAP_PATH):
            self.class_map = np.load(Config.CLASS_MAP_PATH, allow_pickle=True).item()
            self.id_to_class = {v: k for k, v in self.class_map.items()}
            self.num_classes = len(self.class_map)
        else:
            print(
                "Warning: Class map not found. Inference may fail if models rely on it."
            )
            self.class_map = {}
            self.id_to_class = {}
            self.num_classes = 3848  # Default fallback

        # 2. Load Models
        print("Loading Detector...")
        self.detector = CenterNetDetector(pretrained=False)
        if os.path.exists(Config.DETECTOR_CHECKPOINT):
            self.detector.load_state_dict(
                torch.load(Config.DETECTOR_CHECKPOINT, map_location=self.device)
            )
        else:
            print("Warning: Detector checkpoint not found.")
        self.detector.to(self.device)
        self.detector.eval()

        print("Loading Classifier...")
        self.classifier = ResNetClassifier(
            num_classes=self.num_classes, pretrained=False
        )
        if os.path.exists(Config.CLASSIFIER_CHECKPOINT):
            self.classifier.load_state_dict(
                torch.load(Config.CLASSIFIER_CHECKPOINT, map_location=self.device)
            )
        else:
            print("Warning: Classifier checkpoint not found.")
        self.classifier.to(self.device)
        self.classifier.eval()

        # 3. Setup Resize Detector (Cite solution_lesson_node_00021)
        self.resize_detector = ResizeDetector(
            self.detector,
            self.device,
            input_size=Config.DETECTOR_INPUT_SIZE,
            conf_thresh=Config.SCORE_THRESHOLD,
        )

        # 4. Classifier Transform
        self.cls_transform = A.Compose(
            [
                A.Resize(
                    height=Config.CLASSIFIER_INPUT_SIZE,
                    width=Config.CLASSIFIER_INPUT_SIZE,
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )

    def run(
        self,
        test_metadata_path=Config.TEST_METADATA_PATH,
        output_path=Config.SUBMISSION_PATH,
    ):
        if not os.path.exists(test_metadata_path):
            print(f"Test metadata not found at {test_metadata_path}")
            return

        df = pd.read_csv(test_metadata_path)
        results = []

        print(f"Starting inference on {len(df)} images...")

        for idx, row in df.iterrows():
            image_id = row["image_id"]
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            if idx % 50 == 0:
                print(f"Processing image {idx}/{len(df)}")

            # Load Image
            image = cv2.imread(file_path)
            if image is None:
                results.append({"image_id": image_id, "labels": ""})
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 1. Detect
            detections = self.resize_detector.detect(image_rgb)

            if detections.size(0) == 0:
                results.append({"image_id": image_id, "labels": ""})
                continue

            # 2. Global NMS
            keep_indices = nms(
                detections[:, :4], detections[:, 4], Config.NMS_IOU_THRESHOLD
            )
            detections = detections[keep_indices]

            # Limit predictions per page
            if detections.size(0) > Config.MAX_PREDICTIONS_PER_PAGE:
                _, sort_idx = torch.sort(detections[:, 4], descending=True)
                detections = detections[sort_idx[: Config.MAX_PREDICTIONS_PER_PAGE]]

            # 3. Crop & Classify
            crops = []
            valid_dets = []

            # Prepare crops
            for i in range(detections.size(0)):
                x1, y1, x2, y2 = detections[i, :4].int().tolist()

                # Ensure within bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image_rgb.shape[1], x2)
                y2 = min(image_rgb.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image_rgb[y1:y2, x1:x2]

                # Transform
                try:
                    aug = self.cls_transform(image=crop)
                    crops.append(aug["image"])
                    valid_dets.append(detections[i])
                except Exception:
                    continue

            if not crops:
                results.append({"image_id": image_id, "labels": ""})
                continue

            # Batch classify
            crops_tensor = torch.stack(crops).to(self.device)

            cls_preds = []
            # Process in batches
            bs = Config.CLASSIFIER_BATCH_SIZE
            with torch.no_grad():
                for b_i in range(0, len(crops_tensor), bs):
                    batch_crops = crops_tensor[b_i : b_i + bs]
                    outputs = self.classifier(batch_crops)
                    _, preds = torch.max(outputs, 1)
                    cls_preds.extend(preds.cpu().numpy())

            # 4. Format Output
            label_strs = []
            for i, cls_idx in enumerate(cls_preds):
                if cls_idx in self.id_to_class:
                    unicode_char = self.id_to_class[cls_idx]
                else:
                    continue  # Should not happen if map is correct

                det = valid_dets[i]
                x1, y1, x2, y2 = det[:4].tolist()

                # Calculate center
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                label_strs.append(f"{unicode_char} {center_x} {center_y}")

            results.append({"image_id": image_id, "labels": " ".join(label_strs)})

        # Save Submission
        sub_df = pd.DataFrame(results)
        sub_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
