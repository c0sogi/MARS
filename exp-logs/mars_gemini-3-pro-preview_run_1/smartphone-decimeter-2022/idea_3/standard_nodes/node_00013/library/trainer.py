import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.model import GnssDeepSetTCN
from library.utils import set_seed, compute_metric


class Trainer:
    def __init__(self, train_loader=None, val_loader=None, test_loader=None):
        """
        Initializes the Trainer with data loaders and model components.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            test_loader (DataLoader): DataLoader for test data.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = torch.device(Config.DEVICE)
        self.model = GnssDeepSetTCN().to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Mean Absolute Error is robust for GNSS outliers
        self.criterion = nn.L1Loss()

        self.best_score = float("inf")
        self.best_model_path = os.path.join(Config.WORKING_DIR, "model_weights.pth")

        # Ensure reproducibility
        set_seed(Config.SEED)

    def fit(self):
        """
        Runs the full training loop with early stopping.
        """
        if self.train_loader is None or self.val_loader is None:
            raise ValueError(
                "Train and Validation loaders must be provided for fitting."
            )

        print(f"Starting training on device: {self.device}")

        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            # --- Training Phase ---
            self.model.train()
            train_loss = 0.0

            for batch_idx, (sat_feat, glob_feat, mask, target, _) in enumerate(
                self.train_loader
            ):
                # Move to device
                sat_feat = sat_feat.to(self.device)
                glob_feat = glob_feat.to(self.device)
                mask = mask.to(self.device)
                target = target.to(self.device)

                # Reshape for TCN: (Batch, Seq_Len=1, ...)
                # The DataLoader provides independent epochs, so sequence length is 1
                sat_feat = sat_feat.unsqueeze(1)
                glob_feat = glob_feat.unsqueeze(1)
                mask = mask.unsqueeze(1)

                self.optimizer.zero_grad()

                # Forward pass
                # Output shape: (Batch, Seq_Len=1, 2)
                output = self.model(sat_feat, glob_feat, mask)

                # Squeeze sequence dimension to match target: (Batch, 2)
                output = output.squeeze(1)

                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)

            # --- Validation Phase ---
            val_score, val_loss = self.evaluate(self.val_loader)

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Score (Mean 50/95): {val_score:.10f}"
            )

            # --- Checkpointing & Early Stopping ---
            if val_score < self.best_score:
                self.best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  -> New best model saved! Score: {val_score:.10f}")
            else:
                patience_counter += 1
                print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {self.best_score:.10f}")

    def evaluate(self, loader):
        """
        Evaluates the model on the provided loader using the competition metric.

        Args:
            loader (DataLoader): DataLoader to evaluate.

        Returns:
            tuple: (competition_metric_score, average_loss)
        """
        self.model.eval()
        total_loss = 0.0

        all_preds_lat = []
        all_preds_lon = []
        all_gt_lat = []
        all_gt_lon = []

        # We need tripIds and timestamps to group correctly for the metric
        # Since shuffle=False for val/test, we can access the dataset attributes sequentially
        trip_ids = loader.dataset.trip_ids
        timestamps = loader.dataset.timestamps

        # Ensure dataset length matches loader iterations
        # (Drop last batch is False by default in get_dataloaders)

        current_idx = 0

        with torch.no_grad():
            for sat_feat, glob_feat, mask, target, wls_lla in loader:
                batch_size = sat_feat.size(0)

                # Move to device
                sat_feat = sat_feat.to(self.device)
                glob_feat = glob_feat.to(self.device)
                mask = mask.to(self.device)
                target = target.to(self.device)

                # Reshape for TCN
                sat_feat = sat_feat.unsqueeze(1)
                glob_feat = glob_feat.unsqueeze(1)
                mask = mask.unsqueeze(1)

                # Forward
                output = self.model(sat_feat, glob_feat, mask)
                output = output.squeeze(1)  # (Batch, 2)

                # Compute Loss
                loss = self.criterion(output, target)
                total_loss += loss.item() * batch_size

                # Reconstruct Absolute Coordinates
                # Pred = WLS + Residual
                # output[:, 0] is dLat, output[:, 1] is dLon
                # wls_lla[:, 0] is Lat, wls_lla[:, 1] is Lon

                # Move WLS to CPU for numpy operations
                wls_lla_np = wls_lla.numpy()
                output_np = output.cpu().numpy()
                target_np = target.cpu().numpy()

                pred_lat = wls_lla_np[:, 0] + output_np[:, 0]
                pred_lon = wls_lla_np[:, 1] + output_np[:, 1]

                # Reconstruct GT for verification (GT = WLS + Target)
                gt_lat = wls_lla_np[:, 0] + target_np[:, 0]
                gt_lon = wls_lla_np[:, 1] + target_np[:, 1]

                all_preds_lat.extend(pred_lat)
                all_preds_lon.extend(pred_lon)
                all_gt_lat.extend(gt_lat)
                all_gt_lon.extend(gt_lon)

                current_idx += batch_size

        avg_loss = total_loss / len(loader.dataset)

        # Create DataFrame for metric computation
        df_pred = pd.DataFrame(
            {
                "tripId": trip_ids,
                "UnixTimeMillis": timestamps,
                "LatitudeDegrees": all_preds_lat,
                "LongitudeDegrees": all_preds_lon,
            }
        )

        df_gt = pd.DataFrame(
            {
                "tripId": trip_ids,
                "UnixTimeMillis": timestamps,
                "LatitudeDegrees": all_gt_lat,
                "LongitudeDegrees": all_gt_lon,
            }
        )

        score = compute_metric(df_pred, df_gt)
        return score, avg_loss

    def predict(self):
        """
        Generates predictions for the test set using the best saved model.
        Saves the result to Config.SUBMISSION_PATH.
        """
        if self.test_loader is None:
            raise ValueError("Test loader must be provided for prediction.")

        print("Loading best model for inference...")
        if not os.path.exists(self.best_model_path):
            print("Warning: No best model found. Using current weights.")
        else:
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )

        self.model.eval()

        all_preds_lat = []
        all_preds_lon = []

        print("Generating predictions...")
        with torch.no_grad():
            for sat_feat, glob_feat, mask, _, wls_lla in self.test_loader:
                # Move to device
                sat_feat = sat_feat.to(self.device)
                glob_feat = glob_feat.to(self.device)
                mask = mask.to(self.device)

                # Reshape
                sat_feat = sat_feat.unsqueeze(1)
                glob_feat = glob_feat.unsqueeze(1)
                mask = mask.unsqueeze(1)

                # Forward
                output = self.model(sat_feat, glob_feat, mask)
                output = output.squeeze(1)

                # Reconstruct
                wls_lla_np = wls_lla.numpy()
                output_np = output.cpu().numpy()

                pred_lat = wls_lla_np[:, 0] + output_np[:, 0]
                pred_lon = wls_lla_np[:, 1] + output_np[:, 1]

                all_preds_lat.extend(pred_lat)
                all_preds_lon.extend(pred_lon)

        # Create Submission DataFrame
        # Use dataset attributes which preserve order
        submission_df = pd.DataFrame(
            {
                "tripId": self.test_loader.dataset.trip_ids,
                "UnixTimeMillis": self.test_loader.dataset.timestamps,
                "LatitudeDegrees": all_preds_lat,
                "LongitudeDegrees": all_preds_lon,
            }
        )

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Total predictions: {len(submission_df)}")
