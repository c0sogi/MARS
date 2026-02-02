import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.models import CenterNetDetector, CharacterClassifier
from library.dataset import KuzushijiTestDataset, get_class_map, get_transforms
from library.utils import decode_detections


class InferencePipeline:
    """
    End-to-end inference pipeline for Kuzushiji Character Recognition.
    Combines Stage 1 (Detection) and Stage 2 (Classification).
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.detector = None
        self.classifier = None
        self.idx_to_char = None
        self.classifier_transform = None

    def load_resources(self):
        """
        Loads models and mappings.
        """
        print("Loading resources...")

        # 1. Load Class Map
        _, self.idx_to_char = get_class_map(load_cached=True)

        # 2. Load Detector
        self.detector = CenterNetDetector(pretrained=False)
        if os.path.exists(Config.DETECTOR_MODEL_PATH):
            state_dict = torch.load(
                Config.DETECTOR_MODEL_PATH, map_location=self.device
            )
            self.detector.load_state_dict(state_dict)
            print(f"Detector weights loaded from {Config.DETECTOR_MODEL_PATH}")
        else:
            print(
                f"Warning: Detector weights not found at {Config.DETECTOR_MODEL_PATH}"
            )

        self.detector.to(self.device)
        self.detector.eval()

        # 3. Load Classifier
        self.classifier = CharacterClassifier(
            num_classes=Config.NUM_CLASSES, pretrained=False
        )
        if os.path.exists(Config.CLASSIFIER_MODEL_PATH):
            state_dict = torch.load(
                Config.CLASSIFIER_MODEL_PATH, map_location=self.device
            )
            self.classifier.load_state_dict(state_dict)
            print(f"Classifier weights loaded from {Config.CLASSIFIER_MODEL_PATH}")
        else:
            print(
                f"Warning: Classifier weights not found at {Config.CLASSIFIER_MODEL_PATH}"
            )

        self.classifier.to(self.device)
        self.classifier.eval()

        # 4. Classifier Transform (for crops)
        # We use the 'test' split transform which does Resize + Normalize
        self.classifier_transform = get_transforms(
            "classifier", "test", Config.CLASSIFIER_IMG_SIZE
        )

    def correct_coordinates(
        self, detections, orig_shape, input_size=Config.DETECTOR_IMG_SIZE
    ):
        """
        Corrects detection coordinates from the resized/padded input space
        back to the original image space.

        Handles the aspect-ratio preserving resize (LongestMaxSize) + Padding used in dataset.py.

        Args:
            detections: numpy array (K, 6) [x, y, w, h, score, class] in input_size scale.
            orig_shape: tuple (h, w) of the original image.
            input_size: int, size of the model input.

        Returns:
            corrected_dets: numpy array (K, 6) with x, y, w, h in original scale.
        """
        orig_h, orig_w = orig_shape

        # Logic matching Albumentations LongestMaxSize + PadIfNeeded
        scale = input_size / max(orig_h, orig_w)

        scaled_h = int(orig_h * scale)
        scaled_w = int(orig_w * scale)

        # Padding is applied to center the image if using default PadIfNeeded behavior
        # (or typically how it's configured in object detection pipelines).
        # Assuming PadIfNeeded pads equally on sides if not specified otherwise.
        pad_top = (input_size - scaled_h) // 2
        pad_left = (input_size - scaled_w) // 2

        corrected_dets = detections.copy()

        # x_orig = (x_det - pad_left) / scale
        corrected_dets[:, 0] = (detections[:, 0] - pad_left) / scale
        # y_orig = (y_det - pad_top) / scale
        corrected_dets[:, 1] = (detections[:, 1] - pad_top) / scale
        # w_orig = w_det / scale
        corrected_dets[:, 2] = detections[:, 2] / scale
        # h_orig = h_det / scale
        corrected_dets[:, 3] = detections[:, 3] / scale

        return corrected_dets

    def process_crops(self, img_tensor_original, detections):
        """
        Extracts crops from the original image based on detections and prepares batch for classifier.

        Args:
            img_tensor_original: numpy array (H, W, 3) - Original image (RGB).
            detections: numpy array (N, 6) - Detections in original scale [x, y, w, h, ...].

        Returns:
            batch_crops: Tensor (N, 3, 64, 64) or None if no valid crops.
        """
        crops = []
        img_h, img_w = img_tensor_original.shape[:2]

        for det in detections:
            cx, cy, w, h = det[0], det[1], det[2], det[3]

            # Convert center/size to top-left/bottom-right
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            # Clamp to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                # Fallback for degenerate boxes: use a small box around center
                x1 = max(0, int(cx - 16))
                y1 = max(0, int(cy - 16))
                x2 = min(img_w, int(cx + 16))
                y2 = min(img_h, int(cy + 16))

            crop = img_tensor_original[y1:y2, x1:x2]

            # Apply classifier transforms (Resize -> Normalize -> ToTensor)
            if self.classifier_transform:
                transformed = self.classifier_transform(image=crop)
                crops.append(transformed["image"])
            else:
                # Fallback manual transform
                crop = cv2.resize(
                    crop, (Config.CLASSIFIER_IMG_SIZE, Config.CLASSIFIER_IMG_SIZE)
                )
                crop = crop.transpose(2, 0, 1).astype(np.float32) / 255.0
                crops.append(torch.from_numpy(crop))

        if not crops:
            return None

        return torch.stack(crops)

    def run(self):
        """
        Executes the full inference pipeline on the test set and saves the submission.
        """
        self.load_resources()

        # Dataset and Loader
        test_dataset = KuzushijiTestDataset()
        # Batch size 1 for test to handle variable original image sizes easily
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=lambda x: x[0],  # Unwrap batch dimension since batch_size=1
        )

        results = []
        print(f"Starting inference on {len(test_dataset)} images...")

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                if data is None:
                    continue

                image_id = data["image_id"]
                orig_shape = data["orig_shape"]  # (H, W)
                img_input = (
                    data["img"].unsqueeze(0).to(self.device)
                )  # (1, 3, 1024, 1024)

                # --- Stage 1: Detection ---
                outputs = self.detector(img_input)

                # Decode detections
                # Returns (1, K, 6)
                detections = decode_detections(
                    outputs["heatmap"],
                    outputs["size_map"],
                    outputs["offset_map"],
                    K=Config.MAX_PREDICTIONS,
                )

                detections = detections.cpu().numpy()[0]  # (K, 6)

                # Filter by confidence
                mask = detections[:, 4] >= Config.CONF_THRESHOLD
                valid_detections = detections[mask]

                prediction_string = ""

                if len(valid_detections) > 0:
                    # Transform coordinates back to original image space
                    # valid_detections contains [x_center, y_center, w, h, score, class]
                    orig_detections = self.correct_coordinates(
                        valid_detections, orig_shape
                    )

                    # Load original image for cropping
                    # Note: Ideally we would pass the original image through the dataloader,
                    # but KuzushijiTestDataset returns the transformed image.
                    # We need to reload the original for high-quality crops.
                    # Optimization: In a persistent worker setup, we'd cache or return both.
                    # Here we do a quick read.
                    full_path = os.path.join(
                        Config.INPUT_DIR, "test_images", f"{image_id}.jpg"
                    )
                    orig_img = cv2.imread(full_path)
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

                    # --- Stage 2: Classification ---
                    # Prepare crops
                    crop_batch = self.process_crops(orig_img, orig_detections)

                    if crop_batch is not None:
                        crop_batch = crop_batch.to(self.device)

                        # Run classifier in chunks if too many detections to fit in VRAM
                        # A100 is large, but safety first.
                        cls_logits = []
                        chunk_size = 256
                        for k in range(0, len(crop_batch), chunk_size):
                            batch_chunk = crop_batch[k : k + chunk_size]
                            logits_chunk = self.classifier(batch_chunk)
                            cls_logits.append(logits_chunk)

                        cls_logits = torch.cat(cls_logits, dim=0)

                        # Get predicted classes
                        pred_indices = torch.argmax(cls_logits, dim=1).cpu().numpy()

                        # Format output string
                        # Format: "Unicode X Y Unicode X Y ..."
                        # X, Y should be center coordinates
                        labels_list = []
                        for j, idx in enumerate(pred_indices):
                            char_code = self.idx_to_char.get(idx, "U+0000")
                            x_c = int(orig_detections[j, 0])
                            y_c = int(orig_detections[j, 1])
                            labels_list.append(f"{char_code} {x_c} {y_c}")

                        prediction_string = " ".join(labels_list)

                results.append({"image_id": image_id, "labels": prediction_string})

                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1} images")

        # Save Submission
        df_sub = pd.DataFrame(results)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
