import os
import torch
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import LidarDataset
from library.model import BevYolo
from library.utils import sensor_to_world


class Predictor:
    """
    Handles inference and submission generation for the BEV-YOLO model.
    """

    def __init__(self, checkpoint_path=None, device=None):
        self.device = device if device else Config.DEVICE

        # Initialize Model
        self.model = BevYolo().to(self.device)

        # Load Weights
        if checkpoint_path is None:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

        if os.path.exists(checkpoint_path):
            print(f"Loading model weights from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Checkpoint {checkpoint_path} not found. Using initialized weights (random)."
            )

        self.model.eval()

        # Decode Parameters
        self.anchors = torch.tensor(Config.ANCHORS, device=self.device)
        self.stride = 4  # Network output stride
        self.voxel_size = torch.tensor(Config.VOXEL_SIZE, device=self.device)
        self.pc_range = torch.tensor(Config.PC_RANGE, device=self.device)

        # Grid Cache
        self.grid_x = None
        self.grid_y = None

    def _decode_predictions(self, predictions):
        """
        Decodes the output tensor from the model into geometric bounding boxes.

        Args:
            predictions: (B, A, H, W, Attribs)
                Attribs: [obj, dx, dy, dw, dl, z, dh, sin, cos, class_logits...]

        Returns:
            decoded: (B, A, H, W, 9)
                Channels: [x, y, z, w, l, h, yaw, score, class_idx]
        """
        B, A, H, W, _ = predictions.shape

        # 1. Generate Grid
        if self.grid_x is None or self.grid_x.shape != (H, W):
            y_range = torch.arange(H, device=self.device)
            x_range = torch.arange(W, device=self.device)
            self.grid_y, self.grid_x = torch.meshgrid(y_range, x_range, indexing="ij")

        grid_x = self.grid_x.view(1, 1, H, W).expand(B, A, H, W)
        grid_y = self.grid_y.view(1, 1, H, W).expand(B, A, H, W)

        # 2. Extract Outputs
        pred_obj = predictions[..., 0]
        pred_reg = predictions[..., 1:9]
        pred_cls = predictions[..., 9:]

        # 3. Objectness Score
        scores = torch.sigmoid(pred_obj)

        # 4. Center Coordinates (Sensor Frame)
        # x = x_min + (gx + 0.5 + dx) * stride * voxel_x
        dx = pred_reg[..., 0]
        dy = pred_reg[..., 1]

        step_x = self.stride * self.voxel_size[0]
        step_y = self.stride * self.voxel_size[1]

        x = self.pc_range[0] + (grid_x + 0.5 + dx) * step_x
        y = self.pc_range[1] + (grid_y + 0.5 + dy) * step_y

        # 5. Dimensions
        # w = anchor_w * exp(dw)
        dw = pred_reg[..., 2]
        dl = pred_reg[..., 3]

        anchor_w = self.anchors[:, 0].view(1, A, 1, 1)
        anchor_l = self.anchors[:, 1].view(1, A, 1, 1)

        w = anchor_w * torch.exp(dw)
        l = anchor_l * torch.exp(dl)

        # 6. Height & Z
        z = pred_reg[..., 4]
        dh = pred_reg[..., 5]
        h = torch.exp(dh)

        # 7. Yaw
        sin_y = pred_reg[..., 6]
        cos_y = pred_reg[..., 7]
        yaw = torch.atan2(sin_y, cos_y)

        # 8. Classification
        # Use softmax for probabilities
        cls_probs = torch.softmax(pred_cls, dim=-1)
        max_cls_prob, cls_indices = torch.max(cls_probs, dim=-1)

        # Final Confidence = Objectness * Class Probability
        final_scores = scores * max_cls_prob

        # Stack Results
        # [x, y, z, w, l, h, yaw, score, class_idx]
        decoded = torch.stack(
            [x, y, z, w, l, h, yaw, final_scores, cls_indices.float()], dim=-1
        )

        return decoded

    def run_inference(self, batch_size=Config.BATCH_SIZE, load_cached_data=True):
        """
        Runs inference on the test set and generates the submission file.
        """
        print("Initializing Test Dataset...")
        dataset = LidarDataset(split="test", load_cached_data=load_cached_data)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        print(f"Starting inference on {len(dataset)} samples...")
        results = []

        with torch.no_grad():
            for batch_idx, (bev, tokens) in enumerate(dataloader):
                bev = bev.to(self.device)

                # Forward Pass
                preds = self.model(bev)

                # Decode
                decoded = self._decode_predictions(preds)  # (B, A, H, W, 9)

                # Flatten to (B, N, 9)
                B = decoded.shape[0]
                decoded = decoded.view(B, -1, 9)

                # Process each sample in the batch
                for i in range(B):
                    token = tokens[i]
                    sample_preds = decoded[i]

                    # 1. Filter by Confidence
                    mask = sample_preds[:, 7] >= Config.CONF_THRESHOLD
                    valid_preds = sample_preds[mask]

                    if valid_preds.shape[0] == 0:
                        results.append({"Id": token, "PredictionString": ""})
                        continue

                    # 2. Non-Maximum Suppression (NMS)
                    # Convert center (x, y, w, l) to corners (x1, y1, x2, y2)
                    x_c = valid_preds[:, 0]
                    y_c = valid_preds[:, 1]
                    w = valid_preds[:, 3]
                    l = valid_preds[:, 4]

                    x1 = x_c - w / 2
                    y1 = y_c - l / 2
                    x2 = x_c + w / 2
                    y2 = y_c + l / 2

                    boxes_nms = torch.stack([x1, y1, x2, y2], dim=1)
                    scores_nms = valid_preds[:, 7]

                    # Apply NMS
                    keep_indices = torchvision.ops.nms(
                        boxes_nms, scores_nms, Config.NMS_IOU_THRESHOLD
                    )
                    final_preds = valid_preds[keep_indices]

                    # 3. Transform to Global Coordinates
                    # Get calibration
                    ego_pose, calib_sensor = dataset.calib_lookup.get_calibration(token)

                    # Move to CPU/Numpy for geometric utils
                    final_preds_np = final_preds.cpu().numpy()

                    # Transform Centers (x, y, z)
                    centers_sensor = final_preds_np[:, 0:3]
                    centers_global = sensor_to_world(
                        centers_sensor, ego_pose, calib_sensor
                    )

                    # Transform Yaw
                    # Construct direction vectors in sensor frame
                    yaws_sensor = final_preds_np[:, 6]
                    vecs_sensor = np.stack(
                        [
                            np.cos(yaws_sensor),
                            np.sin(yaws_sensor),
                            np.zeros_like(yaws_sensor),
                        ],
                        axis=1,
                    )

                    # Transform vectors to global frame (by transforming two points)
                    origins_sensor = np.zeros_like(vecs_sensor)
                    p1_global = sensor_to_world(origins_sensor, ego_pose, calib_sensor)
                    p2_global = sensor_to_world(vecs_sensor, ego_pose, calib_sensor)
                    vecs_global = p2_global - p1_global

                    yaws_global = np.arctan2(vecs_global[:, 1], vecs_global[:, 0])

                    # 4. Format Prediction String
                    pred_strings = []
                    for j in range(final_preds_np.shape[0]):
                        # Data
                        center = centers_global[j]
                        yaw = yaws_global[j]
                        dims = final_preds_np[j, 3:6]  # w, l, h
                        conf = final_preds_np[j, 7]
                        cls_idx = int(final_preds_np[j, 8])
                        class_name = Config.DETECT_CLASSES[cls_idx]

                        # Format: conf x y z w l h yaw class
                        s = f"{conf:.4f} {center[0]:.4f} {center[1]:.4f} {center[2]:.4f} {dims[0]:.4f} {dims[1]:.4f} {dims[2]:.4f} {yaw:.4f} {class_name}"
                        pred_strings.append(s)

                    results.append(
                        {"Id": token, "PredictionString": " ".join(pred_strings)}
                    )

        # Save Submission
        df_sub = pd.DataFrame(results)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")


def generate_predictions(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Wrapper function to run the prediction pipeline.
    """
    predictor = Predictor()
    predictor.run_inference(batch_size=batch_size, load_cached_data=load_cached_data)
