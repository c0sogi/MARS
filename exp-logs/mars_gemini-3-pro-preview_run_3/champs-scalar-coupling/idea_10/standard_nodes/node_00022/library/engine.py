import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, Standardizer, calculate_log_mae
from library.loader import FlattenedGraphDataset, GraphCollator
from library.model import DirectionalMPNN


class Trainer:
    """
    Manages the training, validation, and evaluation of the DirectionalMPNN.
    """

    def __init__(self, load_cached_data=True):
        self.config = Config()
        self.device = torch.device(self.config.DEVICE)

        # Ensure reproducibility
        seed_everything(self.config.SEED)

        print(f"Initializing Trainer on device: {self.device}")

        # 1. Prepare Data
        print("Loading datasets...")
        self.train_dataset = FlattenedGraphDataset(
            split="train", load_cached_data=load_cached_data
        )
        self.val_dataset = FlattenedGraphDataset(
            split="val", load_cached_data=load_cached_data
        )

        self.collator = GraphCollator()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=self.collator,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=self.collator,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # 2. Setup Standardizer
        # We load the stats computed by DataFactory to initialize the Standardizer
        # without needing to load the raw dataframe again.
        print("Setting up Standardizer...")
        self.standardizer = Standardizer(device=self.device)
        self._load_stats_into_standardizer()

        # 3. Initialize Model
        print("Initializing model...")
        self.model = DirectionalMPNN().to(self.device)

        # 4. Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.SCHEDULER_T_0,
            T_mult=self.config.SCHEDULER_T_MULT,
            eta_min=1e-6,
        )

        # Loss function
        self.criterion = nn.L1Loss()

    def _load_stats_into_standardizer(self):
        """
        Loads statistics from the processed data directory and manually configures
        the Standardizer instance.
        """
        stats_path = os.path.join(self.config.PROCESSED_DATA_DIR, "stats.npy")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Statistics file not found at {stats_path}. Run DataFactory first."
            )

        stats = np.load(stats_path, allow_pickle=True).item()

        # 1. Primary Target Stats (Per Type)
        coupling_stats = stats.get("coupling_stats", {})

        # Populate internal dicts
        for t_idx, s in coupling_stats.items():
            self.standardizer.means[t_idx] = s["mean"]
            self.standardizer.stds[t_idx] = s["std"]

        # Re-create the tensors that Standardizer.fit() usually creates
        num_types = len(self.config.TYPE_MAP)
        mean_tensor = torch.zeros(num_types, device=self.device)
        std_tensor = torch.ones(num_types, device=self.device)

        for idx in range(num_types):
            mean_tensor[idx] = self.standardizer.means.get(idx, 0.0)
            std_tensor[idx] = self.standardizer.stds.get(idx, 1.0)

        self.standardizer.mean_tensor = mean_tensor
        self.standardizer.std_tensor = std_tensor

        # 2. Auxiliary Stats
        s_mean = stats.get("shielding_mean", 0.0)
        s_std = stats.get("shielding_std", 1.0)
        c_mean = stats.get("charge_mean", 0.0)
        c_std = stats.get("charge_std", 1.0)

        self.standardizer.set_aux_stats(s_mean, s_std, c_mean, c_std)
        print("Standardizer configured with loaded statistics.")

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0.0
        total_coupling_loss = 0.0
        total_aux_loss = 0.0

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(batch)

            # --- Calculate Loss ---

            # 1. Primary Task: Coupling Constant
            # Standardize targets
            target_coupling = batch["coupling_value"]
            target_types = batch["coupling_type"]

            if len(target_coupling) > 0:
                target_coupling_z = self.standardizer.transform(
                    target_coupling, target_types
                )
                # Ensure preds are squeezed to match target shape
                loss_coupling = self.criterion(
                    preds["coupling"].view(-1), target_coupling_z.view(-1)
                )
            else:
                loss_coupling = torch.tensor(0.0, device=self.device)

            # 2. Auxiliary Tasks
            # Standardize targets
            target_shield = batch["aux_shield"]
            target_charge = batch["aux_charge"]

            target_shield_z = self.standardizer.transform_aux(
                target_shield, "shielding"
            )
            target_charge_z = self.standardizer.transform_aux(target_charge, "charge")

            loss_shield = self.criterion(
                preds["shielding"].view(-1), target_shield_z.view(-1)
            )
            loss_charge = self.criterion(
                preds["charge"].view(-1), target_charge_z.view(-1)
            )

            loss_aux = loss_shield + loss_charge

            # Total Loss
            loss = loss_coupling + self.config.AUX_LOSS_WEIGHT * loss_aux

            # Backward
            loss.backward()

            # Gradient clipping (optional but recommended for GNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.optimizer.step()

            # Step scheduler (CosineAnnealingWarmRestarts expects step-wise or epoch-wise?
            # Usually epoch-wise, but can be step-wise. We'll do epoch-wise in fit(),
            # or step-wise here if we want finer granularity. Let's stick to epoch-wise for simplicity
            # unless using OneCycleLR. Config says CosineAnnealingWarmRestarts, usually per epoch.)

            total_loss += loss.item()
            total_coupling_loss += loss_coupling.item()
            total_aux_loss += loss_aux.item()

        avg_loss = total_loss / len(self.train_loader)
        avg_coupling = total_coupling_loss / len(self.train_loader)
        avg_aux = total_aux_loss / len(self.train_loader)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch_idx+1} | Train Loss: {avg_loss:.6f} (Coup: {avg_coupling:.6f}, Aux: {avg_aux:.6f}) | Time: {elapsed:.1f}s"
        )

        return avg_loss

    def validate(self):
        self.model.eval()

        all_preds = []
        all_targets = []
        all_types = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device)

                # Forward
                preds = self.model(batch)

                # Get coupling predictions
                pred_coupling_z = preds["coupling"].view(-1)

                # Get metadata
                target_types = batch["coupling_type"]
                target_vals = batch["coupling_value"]

                if len(target_types) == 0:
                    continue

                # Inverse Transform Predictions
                pred_coupling_raw = self.standardizer.inverse_transform(
                    pred_coupling_z, target_types
                )

                all_preds.append(pred_coupling_raw.cpu().numpy())
                all_targets.append(target_vals.cpu().numpy())
                all_types.append(target_types.cpu().numpy())

        if not all_preds:
            return 0.0

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_types = np.concatenate(all_types)

        # Calculate Metric
        avg_log_mae, metrics_per_type = calculate_log_mae(
            all_preds, all_targets, all_types
        )

        print(f"Validation Log MAE: {avg_log_mae:.9f}")
        # Print per type for debugging/insight
        type_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics_per_type.items()])
        print(f"  Types: {type_str}")

        return avg_log_mae

    def fit(self):
        print("Starting training...")
        best_metric = float("inf")
        patience_counter = 0

        for epoch in range(self.config.MAX_EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_metric = self.validate()

            # Scheduler Step
            self.scheduler.step(epoch + 1)

            # Checkpoint & Early Stopping
            # Metric is Log MAE (lower is better)
            if val_metric < best_metric:
                print(
                    f"  [New Best] Log MAE improved from {best_metric:.9f} to {val_metric:.9f}. Saving model..."
                )
                best_metric = val_metric
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{self.config.PATIENCE}"
                )

            if patience_counter >= self.config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Log MAE: {best_metric:.9f}")
