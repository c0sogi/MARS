import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_logger, set_seed
from library.model import DLASeg
from library.dataset import BEVDataset, worker_init_fn

# Initialize logger
logger = get_logger()


def _nms(heat, kernel=3):
    """
    Performs Non-Maximum Suppression on the heatmap using max pooling.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _gather_feat(feat, ind):
    """
    Gathers features from specific indices in the feature map.
    Args:
        feat: (B, C, H, W) or (B, H, W, C) -> flattened to (B, H*W, C)
        ind: (B, K) indices
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat


def _topk(scores, K=40):
    """
    Extracts the top K scores, indices, classes, and coordinates from the heatmap.
    """
    batch, cat, height, width = scores.size()

    # Flatten scores to (B, C * H * W) to find top K across all classes and pixels
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    # Convert flattened index to class, y, x
    topk_clses = (topk_inds // (height * width)).int()
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def _transpose_and_gather_feat(feat, ind):
    """
    Transposes feature map and gathers values at specific indices.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class Predictor:
    def __init__(self, checkpoint_path=None):
        """
        Initialize the predictor with model and device.
        """
        set_seed(Config.SEED)
        self.device = torch.device(Config.DEVICE)

        # Load Model
        logger.info(f"Building model: {Config.BACKBONE}...")
        self.model = DLASeg().to(self.device)

        # Load Weights
        if checkpoint_path is None:
            checkpoint_path = Config.MODEL_SAVE_PATH

        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            logger.warning(
                f"No checkpoint found at {checkpoint_path}. Using random weights (DEBUG mode only)."
            )

        self.model.eval()

    def decode_predictions(self, outputs, K=50, score_thresh=0.2):
        """
        Decodes model outputs into 3D bounding boxes.
        Args:
            outputs: Dictionary of model outputs
            K: Max number of detections
            score_thresh: Minimum score threshold
        Returns:
            detections: List of lists containing detection dicts for each sample in batch
        """
        # 1. Heatmap Processing
        hm = torch.sigmoid(outputs["hm"])
        hm = _nms(hm)

        # 2. Extract Top K
        scores, inds, clses, ys, xs = _topk(hm, K=K)

        # 3. Gather Regression Heads
        # reg: (B, K, 2) - offsets
        reg = _transpose_and_gather_feat(outputs["reg"], inds)
        # wh: (B, K, 3) - dimensions
        wh = _transpose_and_gather_feat(outputs["wh"], inds)
        # depth: (B, K, 1) - z coordinate
        depth = _transpose_and_gather_feat(outputs["depth"], inds)
        # rot: (B, K, 2) - sin, cos
        rot = _transpose_and_gather_feat(outputs["rot"], inds)

        # 4. Convert to World Coordinates
        # Apply offsets to grid coordinates
        xs = xs.view(xs.size(0), K, 1) + reg[:, :, 0:1]
        ys = ys.view(ys.size(0), K, 1) + reg[:, :, 1:2]

        # Scale to world coordinates
        # x_world = x_grid * resolution + x_min
        # y_world = y_grid * resolution + y_min

        # Note: dataset.py uses: x_c = (x - x_min) / res
        # So: x = x_c * res + x_min
        xs_world = xs * Config.BEV_RESOLUTION + Config.X_RANGE[0]
        ys_world = ys * Config.BEV_RESOLUTION + Config.Y_RANGE[0]

        # 5. Process Attributes
        # Rotation: atan2(sin, cos)
        yaw = torch.atan2(rot[:, :, 0:1], rot[:, :, 1:2])

        # 6. Assemble Detections
        batch_size = scores.size(0)
        results = []

        for i in range(batch_size):
            sample_dets = []
            for j in range(K):
                score = scores[i, j].item()
                if score < score_thresh:
                    continue

                cls_id = clses[i, j].item()
                cls_name = Config.ID_TO_CLASS[cls_id]

                det = {
                    "score": score,
                    "center_x": xs_world[i, j].item(),
                    "center_y": ys_world[i, j].item(),
                    "center_z": depth[i, j].item(),
                    "width": wh[i, j, 0].item(),
                    "length": wh[i, j, 1].item(),
                    "height": wh[i, j, 2].item(),
                    "yaw": yaw[i, j].item(),
                    "class_name": cls_name,
                }
                sample_dets.append(det)
            results.append(sample_dets)

        return results

    def format_submission_string(self, detections):
        """
        Formats a list of detection dicts into the submission string format.
        Format: score x y z w l h yaw class_name
        """
        if not detections:
            return ""

        # Sort by score descending
        detections.sort(key=lambda x: x["score"], reverse=True)

        strings = []
        for d in detections:
            # Format: score x y z w l h yaw class_name
            s = (
                f"{d['score']:.4f} {d['center_x']:.2f} {d['center_y']:.2f} {d['center_z']:.2f} "
                f"{d['width']:.2f} {d['length']:.2f} {d['height']:.2f} {d['yaw']:.2f} {d['class_name']}"
            )
            strings.append(s)

        return " ".join(strings)

    def run_inference(self, sample_size=None):
        """
        Runs inference on the test set and generates the submission file.
        """
        logger.info("Starting Inference...")

        # Load Test Dataset
        test_dataset = BEVDataset(
            split="test", load_cached_data=True, sample_size=sample_size
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        )

        logger.info(f"Test samples: {len(test_dataset)}")

        submission_data = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input"].to(self.device)
                sample_tokens = batch["sample_token"]

                outputs = self.model(inputs)

                # Decode
                batch_detections = self.decode_predictions(
                    outputs,
                    K=Config.MAX_DETECTIONS,
                    score_thresh=Config.SCORE_THRESHOLD,
                )

                # Format
                for token, dets in zip(sample_tokens, batch_detections):
                    pred_str = self.format_submission_string(dets)
                    submission_data.append({"Id": token, "PredictionString": pred_str})

        # Create DataFrame
        sub_df = pd.DataFrame(submission_data)

        # Fill NaN/Empty for samples with no predictions (though our logic returns "")
        # Ensure columns are correct
        sub_df = sub_df[["Id", "PredictionString"]]

        # Save
        save_path = Config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sub_df.to_csv(save_path, index=False)

        logger.info(f"Submission saved to {save_path}")
        logger.info(f"Total predictions generated: {len(sub_df)}")


def main():
    # This entry point is for testing the inference module independently if needed
    predictor = Predictor()
    predictor.run_inference()
