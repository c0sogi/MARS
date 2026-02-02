import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import PointPillars, AnchorGenerator
from library.dataset import LyftDataset, collate_fn
from library.utils import setup_logger, decode_boxes, nms_3d


class SubmissionGenerator:
    def __init__(self, model_path=Config.MODEL_SAVE_PATH, device=None):
        """
        Initialize the SubmissionGenerator.

        Args:
            model_path (str): Path to the trained model checkpoint.
            device (torch.device, optional): Device to run inference on.
        """
        Config.set_seed()
        self.logger = setup_logger(os.path.join(Config.WORKING_DIR, "submission.log"))
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_path = model_path

        self.logger.info(f"Initializing SubmissionGenerator on device: {self.device}")

        # Initialize Model Architecture
        self.model = PointPillars().to(self.device)
        self.anchor_generator = AnchorGenerator()

        # Load Weights
        if os.path.exists(self.model_path):
            self.logger.info(f"Loading model weights from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            self.logger.warning(
                f"Checkpoint not found at {self.model_path}. Using random initialization."
            )

        self.model.eval()

    def generate(
        self,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        max_samples=None,
    ):
        """
        Run inference on the test set and generate the submission CSV.

        Args:
            batch_size (int): Batch size for the dataloader.
            num_workers (int): Number of worker threads for data loading.
            max_samples (int, optional): Limit the number of samples processed (for debugging).
        """
        self.logger.info("Loading Test Dataset...")

        # Load test dataset
        # load_cached_data=True allows the dataset to use cached artifacts if available
        test_ds = LyftDataset(
            Config.TEST_METADATA_PATH, mode="test", load_cached_data=True
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        total_samples = len(test_ds)
        if max_samples:
            total_samples = min(total_samples, max_samples)
            self.logger.info(f"Processing limited subset: {total_samples} samples")
        else:
            self.logger.info(f"Processing full test set: {total_samples} samples")

        # Pre-compute anchors and move to device
        anchors = self.anchor_generator.get_anchors().to(self.device)
        results = []
        samples_processed = 0

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if max_samples and samples_processed >= max_samples:
                    break

                # Move inputs to device
                pillars = batch["pillars"].to(self.device)
                coords = batch["pillar_coords"].to(self.device)
                num_points = batch["num_points"].to(self.device)
                sample_tokens = batch["sample_tokens"]

                # Forward Pass
                cls_preds, box_preds, dir_preds = self.model(
                    pillars, coords, num_points
                )

                # cls_preds: (B, N_anchors, NumClasses)
                # box_preds: (B, N_anchors, 7)

                batch_size_actual = cls_preds.shape[0]

                for b in range(batch_size_actual):
                    if max_samples and samples_processed >= max_samples:
                        break

                    token = sample_tokens[b]
                    samples_processed += 1

                    # 1. Get Scores and Labels
                    # Apply sigmoid to convert logits to probabilities
                    scores = torch.sigmoid(cls_preds[b])

                    # Get max score and corresponding class index per anchor
                    max_scores, labels = scores.max(dim=1)  # (N_anchors,)

                    # 2. Filter by Score Threshold
                    # This dramatically reduces the number of boxes for NMS
                    mask = max_scores > Config.SCORE_THRESHOLD

                    if not mask.any():
                        results.append({"Id": token, "PredictionString": ""})
                        continue

                    # 3. Decode Boxes
                    valid_scores = max_scores[mask]
                    valid_labels = labels[mask]
                    valid_box_preds = box_preds[b][mask]
                    valid_anchors = anchors[mask]

                    # Decode regression deltas to absolute coordinates
                    decoded_boxes = decode_boxes(valid_box_preds, valid_anchors)

                    # 4. NMS (CPU)
                    # Move to CPU for numpy-based NMS
                    boxes_np = decoded_boxes.cpu().numpy()
                    scores_np = valid_scores.cpu().numpy()
                    labels_np = valid_labels.cpu().numpy()

                    keep_indices = nms_3d(
                        boxes_np,
                        scores_np,
                        threshold=Config.NMS_IOU_THRESHOLD,
                        max_detections=Config.MAX_DETECTIONS,
                    )

                    # 5. Format Prediction String
                    pred_strings = []
                    for k in keep_indices:
                        box = boxes_np[k]
                        score = scores_np[k]
                        label_idx = labels_np[k]  # 0-based index from model

                        # Map to class name
                        # Config.ID_TO_CLASS is 1-based dictionary
                        class_name = Config.ID_TO_CLASS[label_idx + 1]

                        # Format: score x y z w l h yaw class
                        # box: [x, y, z, w, l, h, yaw]
                        s = (
                            f"{score:.6f} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} "
                            f"{box[3]:.6f} {box[4]:.6f} {box[5]:.6f} {box[6]:.6f} {class_name}"
                        )
                        pred_strings.append(s)

                    prediction_string = " ".join(pred_strings)
                    results.append({"Id": token, "PredictionString": prediction_string})

                # Log progress periodically
                if (i + 1) % 10 == 0:
                    self.logger.info(
                        f"Processed batch {i + 1} ({samples_processed} samples)"
                    )

        # Save to CSV
        df = pd.DataFrame(results)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        df.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        self.logger.info(f"Total predictions generated: {len(df)}")
