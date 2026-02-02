import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time

from library.config import TrainConfig, ModelConfig, DataConfig, VoxelConfig, set_seeds
from library.dataset import LidarDataset, collate_fn
from library.model import CenterPointPillars
from library.loss import CenterPointLoss


class Trainer:
    def __init__(self, debug_subset_size=None):
        set_seeds(TrainConfig.seed)
        self.device = torch.device(TrainConfig.device)
        self.config = TrainConfig
        self.model_config = ModelConfig
        self.voxel_config = VoxelConfig

        # Initialize Model
        self.model = CenterPointPillars().to(self.device)

        # Initialize Loss
        self.criterion = CenterPointLoss().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Data Loaders
        self.train_dataset = LidarDataset(
            metadata_path=DataConfig.train_metadata_path,
            split="train",
            enable_augmentation=DataConfig.enable_augmentation,
            has_targets=True,
            subset_size=debug_subset_size or self.config.debug_subset_size,
        )

        self.val_dataset = LidarDataset(
            metadata_path=DataConfig.val_metadata_path,
            split="val",
            enable_augmentation=False,
            has_targets=True,
            subset_size=debug_subset_size or self.config.debug_subset_size,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=self.config.pct_start,
            div_factor=self.config.div_factor,
            final_div_factor=self.config.final_div_factor,
        )

    def train(self):
        print(f"Starting training on device: {self.device}")
        best_val_loss = float("inf")
        patience_counter = 0
        patience_limit = 5  # Early stopping patience

        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss_sum = 0.0

            start_time = time.time()

            for batch_idx, batch in enumerate(self.train_loader):
                # Move batch to device
                pillar_features = batch["pillar_features"].to(self.device)
                pillar_coords = batch["pillar_coords"].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

                batched_inputs = {
                    "pillar_features": pillar_features,
                    "pillar_coords": pillar_coords,
                    "batch_size": batch["batch_size"],
                }

                self.optimizer.zero_grad()

                # Forward
                preds = self.model(batched_inputs)

                # Loss
                loss_dict = self.criterion(preds, targets)
                loss = loss_dict["loss"]

                # Backward
                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )

                self.optimizer.step()
                self.scheduler.step()

                train_loss_sum += loss.item()

                if (batch_idx + 1) % self.config.log_interval == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    # print(f"Epoch {epoch+1}/{self.config.epochs} | Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

            avg_train_loss = train_loss_sum / len(self.train_loader)

            # Validation
            avg_val_loss = self.validate()

            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch+1} | Time: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss}"
            )

            # Checkpoint & Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.best_model_path)
                # print(f"New best model saved with Val Loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1

            # Save latest
            torch.save(self.model.state_dict(), self.config.latest_model_path)

            if patience_counter >= patience_limit:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print("Training complete.")

    def validate(self):
        self.model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                pillar_features = batch["pillar_features"].to(self.device)
                pillar_coords = batch["pillar_coords"].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

                batched_inputs = {
                    "pillar_features": pillar_features,
                    "pillar_coords": pillar_coords,
                    "batch_size": batch["batch_size"],
                }

                preds = self.model(batched_inputs)
                loss_dict = self.criterion(preds, targets)
                val_loss_sum += loss_dict["loss"].item()

        return val_loss_sum / len(self.val_loader)

    def generate_submission(self):
        print("Generating submission...")

        # Load Best Model
        if os.path.exists(self.config.best_model_path):
            self.model.load_state_dict(
                torch.load(self.config.best_model_path, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: Best model not found, using current model state.")

        self.model.eval()

        # Test Dataset
        test_dataset = LidarDataset(
            metadata_path=DataConfig.test_metadata_path,
            split="test",
            enable_augmentation=False,
            has_targets=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        results = []

        with torch.no_grad():
            for batch in test_loader:
                pillar_features = batch["pillar_features"].to(self.device)
                pillar_coords = batch["pillar_coords"].to(self.device)
                tokens = batch["tokens"]
                matrices = batch["matrices"].numpy()  # (B, 4, 4)

                batched_inputs = {
                    "pillar_features": pillar_features,
                    "pillar_coords": pillar_coords,
                    "batch_size": batch["batch_size"],
                }

                preds = self.model(batched_inputs)

                # Decode predictions
                batch_boxes = self._decode_predictions(preds)

                # Format for submission
                for i, boxes in enumerate(batch_boxes):
                    sample_token = tokens[i]
                    matrix = matrices[i]

                    if len(boxes) == 0:
                        results.append({"Id": sample_token, "PredictionString": ""})
                        continue

                    # Transform boxes to global coordinates
                    # boxes: [x, y, z, w, l, h, yaw, score, class_idx]
                    # matrix: Global -> Sensor. We need Sensor -> Global = inv(matrix)

                    # Compute Sensor -> Global
                    # Note: The matrix provided by dataset is Global -> Sensor.
                    # We need to invert it to go back to global.
                    sens_to_global = np.linalg.inv(matrix)

                    pred_strings = []
                    for box in boxes:
                        x, y, z, w, l, h, yaw, score, cls_idx = box

                        # Transform Center
                        center_sens = np.array([x, y, z, 1.0])
                        center_glob = sens_to_global @ center_sens

                        # Transform Yaw
                        # Rotate a unit vector pointing in yaw direction
                        vec_sens = np.array([np.cos(yaw), np.sin(yaw), 0.0, 0.0])
                        vec_glob = sens_to_global @ vec_sens
                        yaw_glob = np.arctan2(vec_glob[1], vec_glob[0])

                        class_name = self.model_config.class_names[int(cls_idx)]

                        # Format: score x y z w l h yaw class_name
                        # Note: Task description asks for width length height
                        # Our box is w, l, h.
                        s = f"{score:.4f} {center_glob[0]:.4f} {center_glob[1]:.4f} {center_glob[2]:.4f} {w:.4f} {l:.4f} {h:.4f} {yaw_glob:.4f} {class_name}"
                        pred_strings.append(s)

                    results.append(
                        {"Id": sample_token, "PredictionString": " ".join(pred_strings)}
                    )

        # Save CSV
        df_sub = pd.DataFrame(results)
        df_sub.to_csv(self.config.submission_path, index=False)
        print(f"Submission saved to {self.config.submission_path}")

    def _decode_predictions(self, preds, score_thresh=0.1, top_k=50):
        """
        Decode model outputs into bounding boxes.
        Returns list of numpy arrays, one per sample in batch.
        Each array: (N, 9) [x, y, z, w, l, h, yaw, score, class_idx]
        """
        hm = torch.sigmoid(preds["hm"])  # (B, C, H, W)
        center_z = preds["center_z"]  # (B, 1, H, W)
        dim = torch.exp(preds["dim"])  # (B, 3, H, W)
        rot = preds["rot"]  # (B, 2, H, W)
        reg = preds["reg"]  # (B, 2, H, W)

        batch_size, num_classes, H, W = hm.shape

        # 1. Max Pooling (NMS)
        padding = 1
        hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=padding)
        keep = (hmax == hm).float()
        hm = hm * keep

        # 2. Top K
        # Flatten: (B, C*H*W)
        hm_flat = hm.view(batch_size, -1)
        topk_scores, topk_inds = torch.topk(hm_flat, top_k)

        # Unravel indices
        topk_clses = (topk_inds // (H * W)).float()
        topk_inds = topk_inds % (H * W)
        topk_ys = (topk_inds // W).float()
        topk_xs = (topk_inds % W).float()

        # 3. Gather features
        # Helper to gather features at specific indices
        def gather_feat(feat, inds):
            # feat: (B, C, H, W) -> (B, H*W, C)
            feat = feat.permute(0, 2, 3, 1).contiguous()
            feat = feat.view(batch_size, -1, feat.size(3))
            # inds: (B, K) -> (B, K, C)
            inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), feat.size(2))
            return feat.gather(1, inds)

        # Gather
        reg_feat = gather_feat(reg, topk_inds)  # (B, K, 2)
        z_feat = gather_feat(center_z, topk_inds)  # (B, K, 1)
        dim_feat = gather_feat(dim, topk_inds)  # (B, K, 3)
        rot_feat = gather_feat(rot, topk_inds)  # (B, K, 2)

        # 4. Decode
        # Grid to Metric
        # x_metric = (x_grid + reg_x) * stride * voxel_size + min_range
        xs = topk_xs + reg_feat[..., 0]
        ys = topk_ys + reg_feat[..., 1]

        stride = self.config.out_size_factor
        voxel_size = self.voxel_config.voxel_size
        pc_range = self.voxel_config.point_cloud_range

        xs = xs * stride * voxel_size[0] + pc_range[0]
        ys = ys * stride * voxel_size[1] + pc_range[1]

        # Z is direct
        zs = z_feat[..., 0]

        # Dimensions (w, l, h)
        ws = dim_feat[..., 0]
        ls = dim_feat[..., 1]
        hs = dim_feat[..., 2]

        # Rotation
        # rot is (sin, cos)
        yaws = torch.atan2(rot_feat[..., 0], rot_feat[..., 1])

        # 5. Filter by score
        batch_results = []
        for i in range(batch_size):
            scores = topk_scores[i]
            mask = scores > score_thresh

            if mask.sum() == 0:
                batch_results.append(np.zeros((0, 9), dtype=np.float32))
                continue

            res = torch.stack(
                [
                    xs[i][mask],
                    ys[i][mask],
                    zs[i][mask],
                    ws[i][mask],
                    ls[i][mask],
                    hs[i][mask],
                    yaws[i][mask],
                    scores[mask],
                    topk_clses[i][mask],
                ],
                dim=1,
            )

            batch_results.append(res.cpu().numpy())

        return batch_results
