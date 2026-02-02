import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
from library.utils import GeodeticUtils
from library.config import Config


class Trainer:
    """
    Manages the training, validation, and inference processes for the GNSS model.
    """

    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN
        )

        self.best_score = float("inf")
        self.patience_counter = 0

    def custom_loss(self, outputs, targets, decimated_targets, mask):
        """
        Computes the weighted Mean Absolute Error (MAE) loss for the final output
        and auxiliary deep supervision heads.

        Args:
            outputs (list): List of model outputs [head0, head1, head2, head3].
            targets (torch.Tensor): Full resolution target (B, 2, L).
            decimated_targets (list): List of subsampled targets.
            mask (torch.Tensor): Validity mask (B, L).

        Returns:
            torch.Tensor: Weighted scalar loss.
        """
        total_loss = 0

        # Strides corresponding to the heads (Deepest -> Final)
        # head0: Stride 8, head1: Stride 4, head2: Stride 2, head3: Stride 1
        strides = [8, 4, 2, 1]

        # Ensure we have the correct number of outputs and weights
        assert len(outputs) == len(self.config.AUX_LOSS_WEIGHTS)

        # The last element of decimated_targets in the dataset might be the full res one,
        # or we use the main 'targets' for the final head.
        # Based on Dataset implementation, decimated_targets contains [stride8, stride4, stride2, stride1].

        for i, pred in enumerate(outputs):
            weight = self.config.AUX_LOSS_WEIGHTS[i]
            stride = strides[i]

            # Select target
            if i < len(decimated_targets):
                target = decimated_targets[i]
            else:
                # Fallback to slicing full target if list is short (shouldn't happen with current dataset)
                target = targets[:, :, ::stride]

            # Downsample mask for this level
            sub_mask = mask[:, ::stride]

            # Handle potential shape mismatch due to padding/interpolation
            min_len = min(pred.shape[2], target.shape[2], sub_mask.shape[1])
            pred_sliced = pred[:, :, :min_len]
            target_sliced = target[:, :, :min_len]
            mask_sliced = sub_mask[:, :min_len]

            # MAE Loss
            abs_diff = torch.abs(pred_sliced - target_sliced)

            # Apply mask: (B, L) -> (B, 1, L) to broadcast over channels
            masked_diff = abs_diff * mask_sliced.unsqueeze(1)

            # Normalize by number of valid elements
            # Count valid pixels * channels (2)
            valid_elements = mask_sliced.sum() * 2

            if valid_elements > 0:
                loss = masked_diff.sum() / valid_elements
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            total_loss += weight * loss

        return total_loss

    def train_epoch(self, dataloader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            features = batch["features"].to(self.device)
            mask = batch["masks"].to(self.device)
            targets = batch["targets"].to(self.device)

            decimated_targets = [t.to(self.device) for t in batch["decimated_targets"]]

            self.optimizer.zero_grad()

            outputs = self.model(features)

            loss = self.custom_loss(outputs, targets, decimated_targets, mask)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.GRADIENT_CLIP_VAL
            )

            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        return epoch_loss

    def validate(self, dataloader):
        """
        Evaluates the model on the validation set using the competition metric.
        Metric: Mean of (50th percentile error + 95th percentile error) / 2, averaged over phones.
        """
        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                mask = batch["masks"]  # CPU

                # Inference
                outputs = self.model(features)
                final_pred = outputs[-1].cpu().numpy()  # (B, 2, L)

                # Process each sequence in the batch
                for i in range(len(batch["metadata"])):
                    meta = batch["metadata"][i]
                    drive_id = meta["drive_id"]
                    phone_name = meta["phone_name"]

                    # Get valid length
                    valid_len = int(mask[i].sum().item())

                    # Extract predicted residuals (East, North)
                    pred_enu = final_pred[i, :, :valid_len].T  # (L, 2)
                    pred_e = pred_enu[:, 0]
                    pred_n = pred_enu[:, 1]

                    # Extract target residuals for validation scoring
                    # We compare predicted ENU vs Target ENU directly for distance
                    target_enu = batch["targets"][i, :, :valid_len].cpu().numpy().T
                    target_e = target_enu[:, 0]
                    target_n = target_enu[:, 1]

                    # Calculate Euclidean distance in ENU space (approximation of Haversine for small deltas)
                    # This avoids converting back to Lat/Lon just to calc distance again
                    dist_error = np.sqrt(
                        (pred_e - target_e) ** 2 + (pred_n - target_n) ** 2
                    )

                    df_res = pd.DataFrame(
                        {
                            "drive_id": drive_id,
                            "phone_name": phone_name,
                            "dist_error": dist_error,
                        }
                    )
                    results.append(df_res)

        if not results:
            return 0.0

        all_results = pd.concat(results, ignore_index=True)

        # Calculate Metric
        phone_scores = []
        # Group by phone (drive_id + phone_name uniquely identifies a phone run)
        for _, group in all_results.groupby(["drive_id", "phone_name"]):
            p50 = np.percentile(group["dist_error"], 50)
            p95 = np.percentile(group["dist_error"], 95)
            score = (p50 + p95) / 2
            phone_scores.append(score)

        final_metric = np.mean(phone_scores)
        return final_metric

    def fit(self, train_loader, val_loader):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {self.config.EPOCHS} epochs on {self.device}...")

        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(train_loader, epoch)
            val_score = self.validate(val_loader)

            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Score: {val_score:.10f}"
            )

            # Checkpoint & Early Stopping
            if val_score < self.best_score:
                self.best_score = val_score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                print(f"  -> Model saved (Best Score: {self.best_score:.10f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

    def generate_submission(self, test_loader):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        print("Generating submission...")
        self.model.load_state_dict(
            torch.load(self.config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                mask = batch["masks"]

                outputs = self.model(features)
                final_pred = outputs[-1].cpu().numpy()  # (B, 2, L)

                for i in range(len(batch["metadata"])):
                    meta = batch["metadata"][i]
                    drive_id = meta["drive_id"]
                    phone_name = meta["phone_name"]
                    timestamps = meta["timestamps"]
                    wls_pos = meta["wls_pos"]  # (L, 3) Lat, Lon, Alt

                    valid_len = int(mask[i].sum().item())

                    # Extract predictions
                    pred_enu = final_pred[i, :, :valid_len].T
                    pred_e = pred_enu[:, 0]
                    pred_n = pred_enu[:, 1]

                    # Truncate metadata to valid length if necessary
                    # (In test mode, valid_len should match sequence length unless padded)
                    wls_lat = wls_pos[:valid_len, 0]
                    wls_lon = wls_pos[:valid_len, 1]
                    wls_alt = wls_pos[:valid_len, 2]
                    ts = timestamps[:valid_len]

                    # Reconstruct WGS84 Coordinates from WLS + ENU Residuals
                    # Using GeodeticUtils.enu_to_wgs84 for each point
                    # Since the reference point changes per epoch (WLS baseline),
                    # we treat each point's WLS position as the local reference (0,0,0) for that epoch.
                    # Therefore, we just need to convert the predicted ENU offset to Lat/Lon offset
                    # relative to that specific WLS point.

                    # Vectorized approximation for speed:
                    # dLat = dN / R
                    # dLon = dE / (R * cos(lat))
                    R = 6378137.0
                    dLat = np.rad2deg(pred_n / R)
                    dLon = np.rad2deg(pred_e / (R * np.cos(np.deg2rad(wls_lat))))

                    pred_lat = wls_lat + dLat
                    pred_lon = wls_lon + dLon

                    # Create tripId
                    trip_ids = [f"{drive_id}-{phone_name}" for _ in range(valid_len)]

                    df_pred = pd.DataFrame(
                        {
                            "tripId": trip_ids,
                            "UnixTimeMillis": ts,
                            "LatitudeDegrees": pred_lat,
                            "LongitudeDegrees": pred_lon,
                        }
                    )
                    predictions.append(df_pred)

        submission_df = pd.concat(predictions, ignore_index=True)

        # Ensure we match the sample submission format exactly (rows and columns)
        # The test_loader iterates over full drives, but sample_submission might have gaps or specific order.
        # We need to merge with the template.

        sample_sub = pd.read_csv(
            os.path.join(self.config.INPUT_DIR, "sample_submission.csv")
        )

        # Merge on tripId and UnixTimeMillis
        final_sub = sample_sub[["tripId", "UnixTimeMillis"]].merge(
            submission_df, on=["tripId", "UnixTimeMillis"], how="left"
        )

        # Fill missing values with WLS baseline (or linear interpolation) if any
        # For this implementation, we assume coverage is sufficient or handled by previous steps.
        # If model didn't predict (e.g. masked out), we might have NaNs.
        # A simple fallback is to use the WLS values from input if available, but here we just ffill/bfill for safety.
        final_sub["LatitudeDegrees"] = (
            final_sub["LatitudeDegrees"].fillna(method="ffill").fillna(method="bfill")
        )
        final_sub["LongitudeDegrees"] = (
            final_sub["LongitudeDegrees"].fillna(method="ffill").fillna(method="bfill")
        )

        final_sub.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
