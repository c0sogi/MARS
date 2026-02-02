import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from library.config import Config
from library.utils import set_seed, enu_to_wgs84
from library.dataset import get_dataloaders
from library.model import ResUNet1D
from library.loss import DecimatedMAELoss


class Trainer:
    def __init__(self, debug=False):
        self.device = Config.DEVICE
        self.debug = debug
        set_seed(Config.SEED)

        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            debug=self.debug
        )

        # Model
        self.model = ResUNet1D().to(self.device)

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # Loss
        self.criterion = DecimatedMAELoss().to(self.device)

        # Best Score tracking
        self.best_score = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (features, targets, mask, _) in enumerate(self.train_loader):
            features = features.to(self.device).permute(0, 2, 1)  # (B, C, L)
            targets = targets.to(self.device)  # (B, L, 2)
            mask = mask.to(self.device)  # (B, L)

            self.optimizer.zero_grad()

            # Forward
            outputs = self.model(features)  # List of [final, aux3, aux2]

            # Loss
            loss = self.criterion(outputs, targets, mask)

            # Backward
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()

        # Metric: Mean of (p50 + p95) / 2 per phone/trip
        trip_scores = []

        with torch.no_grad():
            for features, targets, mask, metas in self.val_loader:
                features = features.to(self.device).permute(0, 2, 1)
                targets = targets.to(self.device)
                mask = mask.to(self.device)

                # Forward (only need final output for validation)
                outputs = self.model(features)
                final_pred = outputs[0].permute(0, 2, 1)  # (B, L, 2)

                # Calculate errors per sample in batch
                # Each sample corresponds to a trip
                for i in range(features.size(0)):
                    # Get valid length
                    valid_len = mask[i].sum().item()
                    if valid_len == 0:
                        continue

                    # Extract valid predictions and targets
                    pred_valid = final_pred[i, :valid_len, :].cpu().numpy()
                    target_valid = targets[i, :valid_len, :].cpu().numpy()

                    # Calculate Euclidean distance error (in meters)
                    # Target is (East_GT - East_WLS, North_GT - North_WLS)
                    # Pred is (East_Pred - East_WLS, North_Pred - North_WLS)
                    # Error is distance between (East_Pred, North_Pred) and (East_GT, North_GT)
                    diff = pred_valid - target_valid
                    errors = np.sqrt(np.sum(diff**2, axis=1))

                    # Calculate metrics
                    p50 = np.percentile(errors, 50)
                    p95 = np.percentile(errors, 95)
                    score = (p50 + p95) / 2.0
                    trip_scores.append(score)

        if not trip_scores:
            return float("inf")

        return np.mean(trip_scores)

    def fit(self):
        print(f"Starting training on {self.device}...")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_score = self.validate()

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Score (Mean P50/P95): {val_score:.6f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.0f}s"
            )

            # Early Stopping and Checkpointing
            if val_score < self.best_score:
                print(
                    f"  >>> Score Improved ({self.best_score:.6f} -> {val_score:.6f}). Saving model..."
                )
                self.best_score = val_score
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(
                    f"  >>> No improvement. Patience: {self.patience_counter}/{Config.PATIENCE}"
                )

                if self.patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Val Score: {self.best_score:.6f}")

    def predict(self):
        print("Starting inference on test set...")

        # Load best model
        if os.path.exists(Config.MODEL_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_PATH, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()

        results = []

        with torch.no_grad():
            for features, _, mask, metas in self.test_loader:
                features = features.to(self.device).permute(0, 2, 1)
                mask = mask.to(self.device)

                outputs = self.model(features)
                final_pred = outputs[0].permute(0, 2, 1).cpu().numpy()  # (B, L, 2)

                # Process each trip in batch
                for i in range(features.size(0)):
                    valid_len = mask[i].sum().item()
                    if valid_len == 0:
                        continue

                    # Get valid predictions (East, North offsets)
                    pred_enu = final_pred[
                        i, :valid_len, :
                    ]  # (L, 2) -> E, N. U assumed 0.

                    # Get Metadata for reconstruction
                    # metas is a tuple of dicts (one per sample)
                    meta_i = metas[i]

                    trip_ids = meta_i["tripId"]
                    timestamps = meta_i["UnixTimeMillis"]
                    wls_lat = meta_i["Wls_Lat"]
                    wls_lon = meta_i["Wls_Lon"]
                    wls_alt = meta_i["Wls_Alt"]

                    # Ensure lengths match
                    # The collate_fn pads features, but meta arrays are original length
                    assert len(trip_ids) == valid_len

                    # Reconstruct WGS84
                    # Pred is delta_E, delta_N. delta_U = 0
                    pred_lat = []
                    pred_lon = []

                    for t in range(valid_len):
                        de = pred_enu[t, 0]
                        dn = pred_enu[t, 1]
                        du = 0.0

                        ref_lat = wls_lat[t]
                        ref_lon = wls_lon[t]
                        ref_alt = wls_alt[t]

                        lat, lon, _ = enu_to_wgs84(
                            de, dn, du, ref_lat, ref_lon, ref_alt
                        )
                        pred_lat.append(lat)
                        pred_lon.append(lon)

                    # Append to results
                    for t in range(valid_len):
                        results.append(
                            {
                                "tripId": trip_ids[t],
                                "UnixTimeMillis": timestamps[t],
                                "LatitudeDegrees": pred_lat[t],
                                "LongitudeDegrees": pred_lon[t],
                            }
                        )

        # Create DataFrame
        submission_df = pd.DataFrame(results)

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(
            f"Submission saved to {Config.SUBMISSION_PATH} (Rows: {len(submission_df)})"
        )
