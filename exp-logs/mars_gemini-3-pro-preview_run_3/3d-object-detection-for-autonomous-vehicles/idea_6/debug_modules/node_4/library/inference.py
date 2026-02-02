import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import TemporalPointPillars
from library.dataset import NuScenesLidarDataset
from library.utils import format_submission_string


class InferenceEngine:
    """
    Handles inference on the test set for the Temporal PointPillars model.
    """

    def __init__(self, checkpoint_path=None, subset_size=None):
        """
        Args:
            checkpoint_path (str): Path to the model checkpoint. If None, uses Config default.
            subset_size (int): Optional limit on the number of test samples for debugging.
        """
        self.config = Config
        self.device = torch.device(self.config.DEVICE)
        self.subset_size = subset_size
        self.checkpoint_path = checkpoint_path or self.config.MODEL_SAVE_PATH

        # Ensure working directory exists
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

    def _get_test_loader(self):
        """
        Creates the DataLoader for the test set.
        """
        dataset = NuScenesLidarDataset(
            mode="test", subset_size=self.subset_size, load_cached_data=True
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=NuScenesLidarDataset.collate_fn,
            pin_memory=True,
            drop_last=False,
        )
        return loader

    def _decode_predictions(self, preds_dict, k=100):
        """
        Decodes model outputs into 3D bounding boxes.

        Args:
            preds_dict: Dictionary containing model outputs (heatmap, offset, etc.)
            k: Number of top objects to extract per sample.

        Returns:
            batch_boxes: List of np.arrays (N, 7) [x, y, z, w, l, h, yaw]
            batch_scores: List of np.arrays (N,)
            batch_labels: List of lists of class names
        """
        # 1. Extract Heatmap and apply Sigmoid
        hm = torch.sigmoid(preds_dict["heatmap"])  # (B, C, H, W)
        batch, cat, height, width = hm.size()

        # 2. Max Pooling to find peaks (NMS-free approach)
        # padding=1 ensures output size matches input size
        hm_pool = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
        mask = hm_pool == hm
        hm = hm * mask.float()

        # 3. Top-K Selection
        # Flatten: (B, C*H*W)
        scores, inds = torch.topk(hm.view(batch, -1), k)

        # Convert indices to (c, y, x)
        clses = (inds // (height * width)).long()
        inds = inds % (height * width)
        ys = (inds // width).long()
        xs = (inds % width).long()

        # Helper to gather features from regression maps
        def gather_feat(feat, ind):
            # feat: (B, C, H, W)
            dim = feat.size(1)
            feat = feat.view(batch, dim, -1)  # (B, C, H*W)
            feat = feat.permute(0, 2, 1)  # (B, H*W, C)
            # ind: (B, K) -> (B, K, C)
            ind_expanded = ind.unsqueeze(2).expand(batch, k, dim)
            feat = feat.gather(1, ind_expanded)  # (B, K, C)
            return feat

        # Gather regression heads
        # offset: (B, K, 2)
        offset = gather_feat(preds_dict["offset"], inds)
        # height: (B, K, 1)
        z_pred = gather_feat(preds_dict["height"], inds)
        # dim: (B, K, 3) -> log(l, w, h) based on dataset generation
        dim_pred = gather_feat(preds_dict["dim"], inds)
        # rot: (B, K, 2) -> sin, cos
        rot_pred = gather_feat(preds_dict["rot"], inds)

        # 4. Decode to World Coordinates
        voxel_size = self.config.VOXEL_SIZE
        pc_range = self.config.POINT_CLOUD_RANGE

        # Adjust grid coordinates by predicted offset
        xs = xs.float().view(batch, k, 1) + offset[:, :, 0:1]
        ys = ys.float().view(batch, k, 1) + offset[:, :, 1:2]

        # Convert to world coordinates: x_world = x_grid * voxel_x + min_x
        xs = xs * voxel_size[0] + pc_range[0]
        ys = ys * voxel_size[1] + pc_range[1]

        # Z is predicted directly (absolute)
        zs = z_pred

        # Dimensions: exp(log_dim)
        # Dataset stores [log(l), log(w), log(h)]
        dims = torch.exp(dim_pred)
        ls = dims[:, :, 0:1]
        ws = dims[:, :, 1:2]
        hs = dims[:, :, 2:3]

        # Rotation: atan2(sin, cos)
        yaws = torch.atan2(rot_pred[:, :, 0:1], rot_pred[:, :, 1:2])

        # Concatenate: x, y, z, w, l, h, yaw
        # Submission format expects: center_x center_y center_z width length height yaw
        final_box = torch.cat([xs, ys, zs, ws, ls, hs, yaws], dim=2)

        # Move to CPU
        final_box = final_box.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        clses = clses.detach().cpu().numpy()

        batch_boxes = []
        batch_scores = []
        batch_labels = []

        class_names = self.config.CLASS_NAMES

        for b in range(batch):
            batch_boxes.append(final_box[b])
            batch_scores.append(scores[b])
            labels = [class_names[c] for c in clses[b]]
            batch_labels.append(labels)

        return batch_boxes, batch_scores, batch_labels

    def run_inference(self):
        """
        Executes the inference pipeline: loads model, runs prediction, saves CSV.
        """
        print("Initializing Inference Engine...")

        # 1. Load Model
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.checkpoint_path}"
            )

        print(f"Loading model from {self.checkpoint_path}...")
        model = TemporalPointPillars().to(self.device)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        # 2. Load Data
        test_loader = self._get_test_loader()
        print(f"Test dataset size: {len(test_loader.dataset)} samples")

        results = []

        # 3. Inference Loop
        print("Starting inference loop...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move inputs to device
                voxels = batch["voxels"].to(self.device)
                num_points = batch["num_points"].to(self.device)
                coordinates = batch["coordinates"].to(self.device)
                sample_tokens = batch["sample_tokens"]

                # Forward Pass
                batch_size = len(sample_tokens)
                preds = model(voxels, num_points, coordinates, batch_size=batch_size)

                # Decode Predictions
                batch_boxes, batch_scores, batch_labels = self._decode_predictions(
                    preds, k=self.config.POST_MAX_OBJECTS
                )

                # Format for Submission
                for i, token in enumerate(sample_tokens):
                    boxes = batch_boxes[i]
                    scores = batch_scores[i]
                    labels = batch_labels[i]

                    # Convert to space-delimited string
                    pred_str = format_submission_string(
                        boxes,
                        scores,
                        labels,
                        score_thresh=self.config.POST_SCORE_THRESHOLD,
                    )

                    results.append({"Id": token, "PredictionString": pred_str})

        # 4. Save Results
        submission_df = pd.DataFrame(results)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Inference complete. Submission saved to {self.config.SUBMISSION_PATH}")
        print(f"Total predictions: {len(submission_df)}")
