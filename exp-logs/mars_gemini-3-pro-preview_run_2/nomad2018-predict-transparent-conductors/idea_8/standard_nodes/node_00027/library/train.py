import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import CompositionAwareCGCNN
from library.data import CrystalGraphDataset, collate_batch
from library.utils import Standardizer, set_seed


class Trainer:
    """
    Trainer class for the Composition-Aware CGCNN model.
    Handles training loop, validation, early stopping, and prediction generation.
    """

    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = CompositionAwareCGCNN(config).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
        )

        # Loss function (MSE on standardized targets)
        self.criterion = nn.MSELoss()

        # Target Standardizer
        self.target_scaler = Standardizer(device=self.device)
        self.target_scaler_path = os.path.join(config.CACHE_DIR, "target_scaler.npz")

        # Paths
        self.checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    def _calculate_rmsle(self, y_true, y_pred):
        """
        Calculate Column-wise Root Mean Squared Logarithmic Error.
        Ensures inputs are non-negative before log.
        """
        # Clamp to 0 to avoid log of negative numbers if model predicts slightly negative
        y_pred = torch.clamp(y_pred, min=0.0)
        y_true = torch.clamp(y_true, min=0.0)

        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)

        squared_log_error = (log_pred - log_true) ** 2
        mean_squared_log_error = torch.mean(squared_log_error, dim=0)
        rmsle_per_column = torch.sqrt(mean_squared_log_error)

        return rmsle_per_column

    def train(self, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=False):
        """
        Executes the training pipeline with early stopping.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for DataLoaders.
            debug (bool): If True, uses a subset of data for quick debugging.
        """
        set_seed(self.config.SEED)
        print(f"Starting training on device: {self.device}")

        # 1. Load Datasets
        train_dataset = CrystalGraphDataset(
            metadata_path=self.config.TRAIN_METADATA_PATH,
            split="train",
            load_cached_data=True,
            debug=debug,
        )

        val_dataset = CrystalGraphDataset(
            metadata_path=self.config.VAL_METADATA_PATH,
            split="val",
            load_cached_data=True,
            debug=debug,
        )

        # 2. Fit Target Scaler on Training Data
        print("Fitting target standardizer...")
        all_targets = torch.cat([d.y for d in train_dataset], dim=0)
        self.target_scaler.fit(all_targets)
        self.target_scaler.save(self.target_scaler_path)

        # 3. Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
        )

        # 4. Training Loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # --- Training Step ---
            self.model.train()
            train_loss_accum = 0.0

            for batch in train_loader:
                batch = batch.to(self.device)

                # Standardize targets
                targets_norm = self.target_scaler.transform(batch.y)

                self.optimizer.zero_grad()
                preds_norm = self.model(batch)

                loss = self.criterion(preds_norm, targets_norm)
                loss.backward()
                self.optimizer.step()

                train_loss_accum += loss.item() * batch.num_graphs

            avg_train_loss = train_loss_accum / len(train_dataset)

            # --- Validation Step ---
            self.model.eval()
            val_loss_accum = 0.0
            val_rmsle_accum = torch.zeros(2).to(self.device)

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)

                    # Standardize targets for loss calculation
                    targets_norm = self.target_scaler.transform(batch.y)

                    preds_norm = self.model(batch)
                    loss = self.criterion(preds_norm, targets_norm)

                    val_loss_accum += loss.item() * batch.num_graphs

                    # Calculate RMSLE on original scale
                    preds_orig = self.target_scaler.inverse_transform(preds_norm)
                    # batch.y is already in original scale

                    # Accumulate squared log errors for RMSLE calculation
                    # Note: We calculate batch-wise sum of squared log errors here for accuracy
                    # RMSLE = sqrt( mean( (log(p+1) - log(t+1))^2 ) )
                    log_diff_sq = (
                        torch.log1p(torch.clamp(preds_orig, min=0))
                        - torch.log1p(torch.clamp(batch.y, min=0))
                    ) ** 2
                    val_rmsle_accum += torch.sum(log_diff_sq, dim=0)

            avg_val_loss = val_loss_accum / len(val_dataset)

            # Final RMSLE calculation
            val_rmsle_per_col = torch.sqrt(val_rmsle_accum / len(val_dataset))
            avg_val_rmsle = torch.mean(val_rmsle_per_col).item()

            epoch_time = time.time() - start_time

            print(
                f"Epoch {epoch:03d} | Time: {epoch_time:.2f}s | "
                f"Train Loss (MSE): {avg_train_loss:.6f} | "
                f"Val Loss (MSE): {avg_val_loss:.6f} | "
                f"Val RMSLE: {avg_val_rmsle:.6f} "
                f"(Form: {val_rmsle_per_col[0]:.6f}, Gap: {val_rmsle_per_col[1]:.6f})"
            )

            # --- Early Stopping & Checkpointing ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"  -> New best model saved (Val Loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

    def predict(self, batch_size=Config.BATCH_SIZE, debug=False):
        """
        Generates predictions for the test set using the best trained model.
        Saves the result to submission.csv.
        """
        print("Starting prediction...")

        # 1. Load Test Data
        test_dataset = CrystalGraphDataset(
            metadata_path=self.config.TEST_METADATA_PATH,
            split="test",
            load_cached_data=True,
            debug=debug,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
        )

        # 2. Load Resources
        # Load Model
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.checkpoint_path}"
            )

        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )
        self.model.eval()

        # Load Scaler
        if not os.path.exists(self.target_scaler_path):
            raise FileNotFoundError(
                f"Target scaler not found at {self.target_scaler_path}"
            )
        self.target_scaler.load(self.target_scaler_path)

        # 3. Inference Loop
        ids = []
        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)

                # Forward pass
                preds_norm = self.model(batch)

                # Inverse transform to original scale
                preds_orig = self.target_scaler.inverse_transform(preds_norm)

                # Collect results
                ids.extend(batch.id.cpu().numpy().flatten())
                predictions.append(preds_orig.cpu().numpy())

        # 4. Format Submission
        predictions = np.concatenate(predictions, axis=0)

        submission_df = pd.DataFrame(
            {
                "id": ids,
                "formation_energy_ev_natom": predictions[:, 0],
                "bandgap_energy_ev": predictions[:, 1],
            }
        )

        # Ensure correct column order and sorting
        submission_df = submission_df.sort_values("id")

        output_path = os.path.join(self.config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        print(submission_df.head())
