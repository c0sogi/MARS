import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import copy

from library.config import Config
from library.dataset import NuScenesLidarDataset
from library.model import TemporalPointPillars
from library.loss import DetectionLoss
from library.utils import create_gt_database, format_submission_string


class Trainer:
    """
    Manages training, validation, and inference for the Temporal PointPillars model.
    """

    def __init__(self, load_cached_data=True, subset_size=None):
        self.config = Config
        self.device = torch.device(self.config.DEVICE)
        self.subset_size = subset_size
        self.load_cached_data = load_cached_data

        # Ensure working directory exists
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

        # Set seeds
        self._set_seeds()

    def _set_seeds(self):
        seed = self.config.SEED
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _get_dataloader(self, mode):
        dataset = NuScenesLidarDataset(
            mode=mode,
            subset_size=self.subset_size,
            load_cached_data=self.load_cached_data,
        )

        shuffle = mode == "train"
        # For test/val, we want sequential order usually, but val shuffle is fine

        loader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=NuScenesLidarDataset.collate_fn,
            pin_memory=True,
            drop_last=(mode == "train"),
        )
        return loader

    def _decode_detections(self, preds_dict, k=100):
        """
        Decodes model outputs (heatmaps, regression maps) into 3D bounding boxes.

        Args:
            preds_dict: Dictionary containing model outputs
            k: Number of top objects to extract

        Returns:
            batch_boxes: List of np.arrays (N, 7) [x, y, z, w, l, h, yaw]
            batch_scores: List of np.arrays (N,)
            batch_labels: List of lists of class names
        """
        # 1. Extract Heatmap and apply Sigmoid
        hm = torch.sigmoid(preds_dict["heatmap"])  # (B, C, H, W)
        batch, cat, height, width = hm.size()

        # 2. Max Pooling to find peaks (NMS-free approach)
        # padding=1 ensures same size
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

        # Get batch indices for gather
        # We need to gather from regression maps: (B, C_reg, H, W) -> (B, C_reg, H*W)
        # Then gather using inds (B, K)

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
        # dim: (B, K, 3) -> log(l, w, h)
        dim_pred = gather_feat(preds_dict["dim"], inds)
        # rot: (B, K, 2) -> sin, cos
        rot_pred = gather_feat(preds_dict["rot"], inds)

        # 4. Decode to World Coordinates
        # Grid size
        voxel_size = self.config.VOXEL_SIZE
        pc_range = self.config.POINT_CLOUD_RANGE

        xs = xs.float().view(batch, k, 1) + offset[:, :, 0:1]
        ys = ys.float().view(batch, k, 1) + offset[:, :, 1:2]

        # x_world = x_grid * voxel_x + min_x
        xs = xs * voxel_size[0] + pc_range[0]
        ys = ys * voxel_size[1] + pc_range[1]

        # z is absolute
        zs = z_pred

        # Dimensions: exp(log_dim)
        # dim_pred is (l, w, h) based on dataset generation
        # Dataset: reg_dim[k] = [np.log(l), np.log(w), np.log(h)]
        # So exp gives l, w, h
        dims = torch.exp(dim_pred)
        ls = dims[:, :, 0:1]
        ws = dims[:, :, 1:2]
        hs = dims[:, :, 2:3]

        # Rotation: atan2(sin, cos)
        # rot_pred is (sin, cos)
        yaws = torch.atan2(rot_pred[:, :, 0:1], rot_pred[:, :, 1:2])

        # Concatenate: x, y, z, w, l, h, yaw
        # Submission format expects: center_x center_y center_z width length height yaw
        # Note: Dataset stores w, l, h. Submission expects width, length, height.
        # We have ls, ws, hs.
        # Check submission format in description: "width length height"
        # Check dataset generation: "width length height" -> "l, w, h" in log?
        # Dataset.py: reg_dim[k] = [np.log(l), np.log(w), np.log(h)]
        # So index 0 is length, 1 is width, 2 is height.
        # We need width, length, height.

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

    def fit(self):
        print("Starting training process...")

        # 0. Prepare Augmentation Database
        if self.config.AUG_USE_GT_SAMPLING:
            print("Checking/Generating GT Database...")
            create_gt_database(
                self.config.TRAIN_METADATA_PATH, load_cached_data=self.load_cached_data
            )

        # 1. Data Loaders
        train_loader = self._get_dataloader("train")
        val_loader = self._get_dataloader("val")

        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # 2. Model & Loss
        model = TemporalPointPillars().to(self.device)
        criterion = DetectionLoss(self.config).to(self.device)

        # 3. Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # One Cycle Scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=self.config.EPOCHS,
            pct_start=0.3,
            div_factor=10,
            final_div_factor=100,
        )

        # 4. Training Loop
        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0

        for epoch in range(1, self.config.EPOCHS + 1):
            model.train()
            train_loss_sum = 0.0
            start_time = time.time()

            # --- TRAIN ---
            for i, batch in enumerate(train_loader):
                # Move data to device
                voxels = batch["voxels"].to(self.device)
                num_points = batch["num_points"].to(self.device)
                coordinates = batch["coordinates"].to(self.device)

                targets = batch["targets"]
                # Move targets to device
                for k, v in targets.items():
                    if isinstance(v, torch.Tensor):
                        targets[k] = v.to(self.device)

                optimizer.zero_grad()

                preds = model(voxels, num_points, coordinates)

                loss, loss_dict = criterion(preds, targets)

                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.GRAD_CLIP_NORM
                )

                optimizer.step()
                scheduler.step()

                train_loss_sum += loss.item()

            avg_train_loss = train_loss_sum / len(train_loader)

            # --- VALIDATION ---
            model.eval()
            val_loss_sum = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    voxels = batch["voxels"].to(self.device)
                    num_points = batch["num_points"].to(self.device)
                    coordinates = batch["coordinates"].to(self.device)

                    targets = batch["targets"]
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(self.device)

                    preds = model(voxels, num_points, coordinates)
                    loss, _ = criterion(preds, targets)
                    val_loss_sum += loss.item()

            avg_val_loss = val_loss_sum / len(val_loader)
            epoch_time = time.time() - start_time

            print(
                f"Epoch {epoch}/{self.config.EPOCHS} | "
                f"Train Loss: {avg_train_loss} | "
                f"Val Loss: {avg_val_loss} | "
                f"Time: {epoch_time:.2f}s"
            )

            # --- CHECKPOINTING & EARLY STOPPING ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.config.MODEL_SAVE_PATH)
                print(f"New best model saved to {self.config.MODEL_SAVE_PATH}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered after {patience} epochs of no improvement."
                    )
                    break

        print("Training complete.")

    def predict_and_submit(self):
        print("Starting inference on Test set...")

        # Load Model
        if not os.path.exists(self.config.MODEL_SAVE_PATH):
            print("No model checkpoint found! Cannot predict.")
            return

        model = TemporalPointPillars().to(self.device)
        model.load_state_dict(
            torch.load(self.config.MODEL_SAVE_PATH, map_location=self.device)
        )
        model.eval()

        # Load Test Data
        test_loader = self._get_dataloader("test")

        results = []

        with torch.no_grad():
            for batch in test_loader:
                voxels = batch["voxels"].to(self.device)
                num_points = batch["num_points"].to(self.device)
                coordinates = batch["coordinates"].to(self.device)
                sample_tokens = batch["sample_tokens"]

                # Forward
                preds = model(voxels, num_points, coordinates)

                # Decode
                # Use top 100 predictions per sample
                batch_boxes, batch_scores, batch_labels = self._decode_detections(
                    preds, k=self.config.POST_MAX_OBJECTS
                )

                # Format
                for i, token in enumerate(sample_tokens):
                    boxes = batch_boxes[i]
                    scores = batch_scores[i]
                    labels = batch_labels[i]

                    pred_str = format_submission_string(
                        boxes,
                        scores,
                        labels,
                        score_thresh=self.config.POST_SCORE_THRESHOLD,
                    )

                    results.append({"Id": token, "PredictionString": pred_str})

        # Save Submission
        submission_df = pd.DataFrame(results)
        # Ensure order matches sample_submission if possible, but Id matching is key
        submission_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
