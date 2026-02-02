import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import GroupScaler, SimpleScaler, AverageMeter, mean_log_mae
from library.model import DualGraphNetwork


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the Scalar Coupling Prediction task.
    """

    def __init__(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = DualGraphNetwork().to(self.device)

        # Initialize Optimizer
        # Using AdamW as specified
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        # Using OneCycleLR with parameters from Config
        if self.train_loader is not None:
            total_steps = len(self.train_loader) * Config.MAX_EPOCHS
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=Config.LEARNING_RATE,
                total_steps=total_steps,
                pct_start=Config.PCT_START,
                div_factor=Config.DIV_FACTOR,
                final_div_factor=Config.FINAL_DIV_FACTOR,
            )
        else:
            self.scheduler = None

        # Initialize Scalers
        self.coupling_scaler = GroupScaler()
        self.shield_scaler = SimpleScaler()
        self.charge_scaler = SimpleScaler()

        # Precomputed tensors for fast GPU lookup
        self.coupling_means = None
        self.coupling_stds = None

        # Initialize scalers if training data is available
        if self.train_loader is not None:
            self._initialize_scalers()

    def _initialize_scalers(self):
        """
        Fits or loads scalers for target standardization.
        Constructs GPU-ready tensors for efficient batch processing.
        """
        print("Initializing target scalers...")

        # Access the underlying InMemoryDataset data
        # This allows us to compute global stats without iterating the loader
        data = self.train_loader.dataset.data

        # 1. Coupling Constant Scaler (Grouped by Type)
        # We need to map integer type indices back to strings for the GroupScaler
        types_int = data.coupling_type.numpy()
        types_str = [Config.COUPLING_TYPES[i] for i in types_int]

        # Create a temporary DataFrame for the GroupScaler API
        df_temp = pd.DataFrame(
            {"target": data.coupling_value.numpy(), "type": types_str}
        )

        # Fit or load from cache
        self.coupling_scaler.fit_or_load(df_temp, "target", "type", Config.STATS_PATH)

        # 2. Auxiliary Scalers (Global)
        self.shield_scaler.fit(data.y_shielding.numpy())
        self.charge_scaler.fit(data.y_mulliken.numpy())

        # 3. Prepare GPU Tensors for Coupling Scaler
        # This avoids string lookups inside the training loop
        means_list = [
            self.coupling_scaler.means.get(t, 0.0) for t in Config.COUPLING_TYPES
        ]
        stds_list = [
            self.coupling_scaler.stds.get(t, 1.0) for t in Config.COUPLING_TYPES
        ]

        self.coupling_means = torch.tensor(
            means_list, device=self.device, dtype=torch.float32
        )
        self.coupling_stds = torch.tensor(
            stds_list, device=self.device, dtype=torch.float32
        )

        print("Scalers initialized successfully.")

    def _transform_coupling(self, values, type_indices):
        """
        Standardizes coupling constants using precomputed GPU tensors.
        z = (x - mu) / sigma
        """
        mu = self.coupling_means[type_indices]
        sigma = self.coupling_stds[type_indices]
        return (values - mu) / sigma

    def _inverse_transform_coupling(self, values, type_indices):
        """
        Reverts standardization for coupling constants.
        x = z * sigma + mu
        """
        mu = self.coupling_means[type_indices]
        sigma = self.coupling_stds[type_indices]
        return (values * sigma) + mu

    def train_epoch(self, epoch):
        """
        Executes one training epoch.
        """
        self.model.train()
        meter_loss = AverageMeter()

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward Pass
            pred_c, pred_s, pred_m = self.model(batch)

            # --- Target Standardization ---
            # Primary Target: Coupling Constant (Per-Group)
            target_c = self._transform_coupling(
                batch.coupling_value, batch.coupling_type
            )

            # Aux Target: Shielding (Global)
            # (val - mean) / std
            target_s = (
                batch.y_shielding - self.shield_scaler.mean
            ) / self.shield_scaler.std

            # Aux Target: Mulliken Charge (Global)
            target_m = (
                batch.y_mulliken - self.charge_scaler.mean
            ) / self.charge_scaler.std

            # --- Loss Calculation ---
            loss_c = nn.functional.mse_loss(pred_c, target_c)
            loss_s = nn.functional.mse_loss(pred_s, target_s)
            loss_m = nn.functional.mse_loss(pred_m, target_m)

            # Composite Loss
            total_loss = loss_c + Config.AUX_LOSS_WEIGHT * (loss_s + loss_m)

            # Backward Pass
            total_loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            meter_loss.update(total_loss.item(), batch.num_graphs)

        return meter_loss.avg

    def evaluate(self, loader):
        """
        Evaluates the model and computes the LogMAE metric.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        all_types = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                # Forward Pass (Auxiliary heads ignored for validation metric)
                pred_c, _, _ = self.model(batch)

                # Inverse Transform to Original Scale
                pred_orig = self._inverse_transform_coupling(
                    pred_c, batch.coupling_type
                )

                # Store results
                all_preds.append(pred_orig.cpu())
                all_targets.append(batch.coupling_value.cpu())

                # Convert type indices to strings for metric calculation
                type_indices = batch.coupling_type.cpu().numpy()
                type_strs = [Config.COUPLING_TYPES[i] for i in type_indices]
                all_types.extend(type_strs)

        # Concatenate all batches
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()
        types = np.array(all_types)

        # Compute Metric
        score = mean_log_mae(y_true, y_pred, types)
        return score

    def run(self):
        """
        Main training loop with validation and best model saving.
        """
        print(f"Starting training on device: {self.device}")
        best_score = float("inf")

        for epoch in range(Config.MAX_EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_score = self.evaluate(self.val_loader)

            print(
                f"Epoch {epoch+1}/{Config.MAX_EPOCHS} | Train Loss: {train_loss:.6f} | Val LogMAE: {val_score:.9f}"
            )

            # Save Best Model
            if val_score < best_score:
                best_score = val_score
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  New best model saved! Score: {best_score:.9f}")

    def predict_and_submit(self):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("Loading best model for inference...")
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

        self.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        ids = []
        preds = []

        print("Generating predictions on test set...")
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)

                pred_c, _, _ = self.model(batch)

                # Inverse Transform
                pred_orig = self._inverse_transform_coupling(
                    pred_c, batch.coupling_type
                )

                ids.extend(batch.coupling_id.cpu().numpy())
                preds.extend(pred_orig.cpu().numpy())

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id": ids, "scalar_coupling_constant": preds})

        # Sort by ID (standard practice)
        df_sub.sort_values("id", inplace=True)

        # Save
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
