import os
import torch
import numpy as np
import pandas as pd
import torchvision
from library.config import Config
from library.model import PointPillars
from library.data_loader import create_data_loaders
from library.geometry import DatasetPreprocessor, apply_transform_to_boxes
from library.anchors import AnchorGenerator


class Inference:
    def __init__(self, config=None, checkpoint_path=None):
        self.config = config if config is not None else Config
        self.device = torch.device(self.config.DEVICE)

        # Initialize Model
        self.model = PointPillars(self.config).to(self.device)

        # Load Checkpoint
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.config.WORKING_DIR, "idea_1", "best_model.pth"
            )

        if os.path.exists(checkpoint_path):
            print(f"Loading model from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights."
            )

        self.model.eval()

        # Helpers
        self.anchor_generator = AnchorGenerator(self.config)
        self.preprocessor = DatasetPreprocessor(self.config.TEST_DATA_DIR)

        # Parameters
        self.conf_threshold = 0.1
        self.nms_iou_threshold = 0.5
        self.max_detections = 500

    def decode_predictions(self, cls_preds, reg_preds):
        """
        Decodes model output into 3D bounding boxes.
        Args:
            cls_preds: (B, H, W, Num_Anchors, Num_Classes)
            reg_preds: (B, H, W, Num_Anchors, 7)
        Returns:
            List of dictionaries containing 'boxes', 'scores', 'labels' for each batch item.
        """
        batch_size = cls_preds.shape[0]
        feature_map_size = (cls_preds.shape[1], cls_preds.shape[2])

        # Generate Anchors: (H, W, Num_Anchors_Per_Loc, 7)
        anchors = self.anchor_generator.generate(feature_map_size, device=self.device)
        # Flatten: (N_a, 7)
        anchors = anchors.reshape(-1, 7)

        decoded_batch = []

        for b in range(batch_size):
            # Flatten predictions for this sample
            # cls: (N_a, Num_Classes)
            cur_cls = cls_preds[b].reshape(-1, self.config.NUM_CLASSES)
            # reg: (N_a, 7)
            cur_reg = reg_preds[b].reshape(-1, 7)

            # 1. Sigmoid & Threshold
            probs = torch.sigmoid(cur_cls)
            max_probs, labels = torch.max(probs, dim=1)

            mask = max_probs > self.conf_threshold

            if not mask.any():
                decoded_batch.append(
                    {
                        "boxes": torch.zeros((0, 7), device=self.device),
                        "scores": torch.zeros((0,), device=self.device),
                        "labels": torch.zeros(
                            (0,), device=self.device, dtype=torch.long
                        ),
                    }
                )
                continue

            # Filter
            scores = max_probs[mask]
            pred_labels = labels[mask]
            pred_reg = cur_reg[mask]
            pred_anchors = anchors[mask]

            # 2. Decode Boxes
            # Anchors: x, y, z, w, l, h, yaw
            # Regs: dx, dy, dz, dw, dl, dh, dyaw

            # Dimensions
            a_dims = pred_anchors[:, 3:6]
            a_d = torch.sqrt(a_dims[:, 0] ** 2 + a_dims[:, 1] ** 2)
            a_h = a_dims[:, 2]

            # Center
            x = pred_reg[:, 0] * a_d + pred_anchors[:, 0]
            y = pred_reg[:, 1] * a_d + pred_anchors[:, 1]
            z = pred_reg[:, 2] * a_h + pred_anchors[:, 2]

            # Dimensions (exp)
            w = torch.exp(pred_reg[:, 3]) * a_dims[:, 0]
            l = torch.exp(pred_reg[:, 4]) * a_dims[:, 1]
            h = torch.exp(pred_reg[:, 5]) * a_dims[:, 2]

            # Yaw
            # dyaw = sin(gt - anchor) -> gt = anchor + asin(dyaw)
            dyaw = torch.clamp(pred_reg[:, 6], -0.99, 0.99)
            yaw = pred_anchors[:, 6] + torch.asin(dyaw)

            boxes = torch.stack([x, y, z, w, l, h, yaw], dim=1)

            decoded_batch.append(
                {"boxes": boxes, "scores": scores, "labels": pred_labels}
            )

        return decoded_batch

    def apply_nms(self, decoded_data):
        """
        Applies NMS to decoded boxes.
        """
        final_results = []

        for sample in decoded_data:
            boxes = sample["boxes"]  # (M, 7)
            scores = sample["scores"]
            labels = sample["labels"]

            if boxes.shape[0] == 0:
                final_results.append(sample)
                continue

            # Convert to BEV (Axis Aligned approximation for NMS)
            # Box: x, y, z, w, l, h, yaw
            # Config: w is y-size, l is x-size
            x = boxes[:, 0]
            y = boxes[:, 1]
            w = boxes[:, 3]  # y-dim
            l = boxes[:, 4]  # x-dim

            x1 = x - l / 2
            y1 = y - w / 2
            x2 = x + l / 2
            y2 = y + w / 2

            bev_boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # Apply NMS
            keep_indices = torchvision.ops.nms(
                bev_boxes, scores, self.nms_iou_threshold
            )

            if keep_indices.shape[0] > self.max_detections:
                keep_indices = keep_indices[: self.max_detections]

            final_results.append(
                {
                    "boxes": boxes[keep_indices],
                    "scores": scores[keep_indices],
                    "labels": labels[keep_indices],
                }
            )

        return final_results

    def transform_to_global(self, boxes_sensor, token):
        """
        Transforms boxes from Sensor frame to Global frame.
        boxes_sensor: numpy array (N, 7)
        """
        if len(boxes_sensor) == 0:
            return boxes_sensor

        # Get transforms
        transforms = self.preprocessor.get_sensor_transforms(token)
        if transforms is None:
            return boxes_sensor

        t_global_sensor, _, _ = transforms

        # We need Sensor -> Global, which is inverse of Global -> Sensor
        t_sensor_global = np.linalg.inv(t_global_sensor)

        boxes_global = apply_transform_to_boxes(boxes_sensor, t_sensor_global)
        return boxes_global

    def format_prediction_string(self, boxes, scores, labels):
        """
        Formats predictions into the submission string format.
        score x y z w l h yaw class_name
        """
        if len(boxes) == 0:
            return ""

        parts = []
        # Sort by score descending
        indices = np.argsort(-scores)

        for i in indices:
            score = scores[i]
            box = boxes[i]
            label_idx = labels[i]
            class_name = self.config.CLASS_NAMES[label_idx]

            # Box: x, y, z, w, l, h, yaw
            # Format: score x y z w l h yaw class_name
            # Note: Ensure w, l, h mapping is consistent with competition expectation
            # Competition: width, length, height
            # Our config: w (y), l (x), h (z).
            # Usually 'width' is smaller dimension, 'length' is larger (along heading).
            # In NuScenes, 'width' is y-axis size, 'length' is x-axis size in object frame.
            # We keep our order: w, l, h

            s = f"{score:.4f} {box[0]:.4f} {box[1]:.4f} {box[2]:.4f} {box[3]:.4f} {box[4]:.4f} {box[5]:.4f} {box[6]:.4f} {class_name}"
            parts.append(s)

        return " ".join(parts)

    def run(self, load_cached_data=True):
        print("Starting Inference...")

        # Create Test Loader
        loaders = create_data_loaders(self.config, load_cached_data=load_cached_data)
        test_loader = loaders.get("test")

        if test_loader is None:
            print("Test loader not found.")
            return

        results = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                points = [p.to(self.device) for p in batch["points"]]
                tokens = batch["tokens"]

                # Forward
                cls_preds, reg_preds = self.model(points)

                if cls_preds is None:
                    # Handle empty batch result
                    for token in tokens:
                        results.append({"Id": token, "PredictionString": ""})
                    continue

                # Decode
                decoded = self.decode_predictions(cls_preds, reg_preds)

                # NMS
                post_processed = self.apply_nms(decoded)

                # Transform and Format
                for i, res in enumerate(post_processed):
                    token = tokens[i]

                    boxes_sensor = res["boxes"].cpu().numpy()
                    scores = res["scores"].cpu().numpy()
                    labels = res["labels"].cpu().numpy()

                    # Transform to Global
                    boxes_global = self.transform_to_global(boxes_sensor, token)

                    # Format
                    pred_str = self.format_prediction_string(
                        boxes_global, scores, labels
                    )

                    results.append({"Id": token, "PredictionString": pred_str})

                if (batch_idx + 1) % 50 == 0:
                    print(f"Processed {batch_idx + 1} batches...")

        # Save Submission
        df_sub = pd.DataFrame(results)

        # Ensure all test IDs are present (fill missing with empty)
        # Load sample submission to get all IDs
        sample_sub_path = os.path.join(self.config.INPUT_DIR, "sample_submission.csv")
        if os.path.exists(sample_sub_path):
            df_sample = pd.read_csv(sample_sub_path)
            # Merge to ensure order and completeness
            df_final = df_sample[["Id"]].merge(df_sub, on="Id", how="left")
            df_final["PredictionString"] = df_final["PredictionString"].fillna("")
        else:
            df_final = df_sub

        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)
        df_final.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")


def generate_submission(load_cached_data=True):
    inference = Inference()
    inference.run(load_cached_data=load_cached_data)
