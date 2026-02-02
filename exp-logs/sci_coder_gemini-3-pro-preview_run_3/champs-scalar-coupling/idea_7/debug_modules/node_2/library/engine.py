import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time
import json
from torch_geometric.loader import DataLoader
from library.config import Config
from library.dataset import MolecularGraphDataset
from library.model import SGLGN
from library.utils import AverageMeter, compute_log_mae, TargetStandardizer


class Trainer:
    def __init__(self):
        self.device = Config.DEVICE
        Config.set_seed(Config.SEED)

        # =========================
        # Data Loading
        # =========================
        print("Initializing Datasets...")
        self.train_dataset = MolecularGraphDataset(mode="train", load_cached_data=True)
        self.val_dataset = MolecularGraphDataset(mode="val", load_cached_data=True)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # =========================
        # Target Standardization
        # =========================
        print("Fitting Target Standardizer...")
        self.standardizer = TargetStandardizer()
        # Load train metadata to fit standardizer
        df_train = pd.read_csv(Config.TRAIN_METADATA)
        self.standardizer.fit(df_train)

        # Save stats for inference
        stats_path = os.path.join(Config.WORKING_DIR, "target_stats.json")
        self.standardizer.save(stats_path)

        # Pre-compute standardization tensors for the GPU
        self.type_means = torch.zeros(len(Config.COUPLING_TYPES), device=self.device)
        self.type_stds = torch.zeros(len(Config.COUPLING_TYPES), device=self.device)

        for t_name, t_stats in self.standardizer.stats.items():
            if t_name in Config.COUPLING_TYPE_MAP:
                idx = Config.COUPLING_TYPE_MAP[t_name]
                self.type_means[idx] = t_stats["mean"]
                self.type_stds[idx] = t_stats["std"]

        # Compute Aux Stats (Approximate from subset for speed)
        print("Computing Auxiliary Stats...")
        self.aux_stats = self._compute_aux_stats()

        # =========================
        # Model & Optimizer
        # =========================
        print("Initializing Model...")
        self.model = SGLGN().to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )

        self.criterion = nn.MSELoss()

    def _compute_aux_stats(self, num_batches=50):
        """
        Computes mean and std for auxiliary targets using a subset of data.
        """
        shield_sum = torch.zeros(9, device=self.device)
        shield_sq_sum = torch.zeros(9, device=self.device)
        charge_sum = torch.zeros(1, device=self.device)
        charge_sq_sum = torch.zeros(1, device=self.device)
        count = 0

        # Iterate over a subset
        for i, batch in enumerate(self.train_loader):
            if i >= num_batches:
                break

            b_shield = batch.y_shield.to(self.device)
            b_charge = batch.y_charge.to(self.device)

            # Shielding
            shield_sum += b_shield.sum(dim=0)
            shield_sq_sum += (b_shield**2).sum(dim=0)

            # Charges
            charge_sum += b_charge.sum()
            charge_sq_sum += (b_charge**2).sum()

            count += batch.num_atoms

        if count == 0:
            return {
                "shield_mean": torch.zeros(9, device=self.device),
                "shield_std": torch.ones(9, device=self.device),
                "charge_mean": torch.zeros(1, device=self.device),
                "charge_std": torch.ones(1, device=self.device),
            }

        shield_mean = shield_sum / count
        shield_std = torch.sqrt((shield_sq_sum / count) - (shield_mean**2))
        shield_std[shield_std < 1e-6] = 1.0  # Avoid div by zero

        charge_mean = charge_sum / count
        charge_std = torch.sqrt((charge_sq_sum / count) - (charge_mean**2))
        charge_std[charge_std < 1e-6] = 1.0

        return {
            "shield_mean": shield_mean,
            "shield_std": shield_std,
            "charge_mean": charge_mean,
            "charge_std": charge_std,
        }

    def train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        coup_loss_meter = AverageMeter()
        aux_loss_meter = AverageMeter()

        start_time = time.time()

        for i, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward Pass
            preds, pred_shield, pred_charge = self.model(batch)

            # --- Primary Loss (Coupling) ---
            # Standardize targets
            means = self.type_means[batch.target_type]
            stds = self.type_stds[batch.target_type]
            y_target_norm = (batch.y - means) / stds

            loss_coupling = self.criterion(preds, y_target_norm)

            # --- Auxiliary Loss ---
            # Standardize aux targets
            y_shield_norm = (
                batch.y_shield - self.aux_stats["shield_mean"]
            ) / self.aux_stats["shield_std"]
            y_charge_norm = (
                batch.y_charge - self.aux_stats["charge_mean"]
            ) / self.aux_stats["charge_std"]

            loss_shield = self.criterion(pred_shield, y_shield_norm)
            loss_charge = self.criterion(pred_charge, y_charge_norm)

            loss_aux = loss_shield + loss_charge

            # --- Total Loss ---
            loss = loss_coupling + Config.AUX_LOSS_WEIGHT * loss_aux

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update meters
            loss_meter.update(loss.item(), batch.num_graphs)
            coup_loss_meter.update(loss_coupling.item(), batch.num_graphs)
            aux_loss_meter.update(loss_aux.item(), batch.num_graphs)

        # Step Scheduler
        self.scheduler.step()

        return loss_meter.avg, coup_loss_meter.avg, aux_loss_meter.avg

    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []
        all_types = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                # Forward
                preds_norm, _, _ = self.model(batch)

                # Inverse Transform Predictions
                preds_orig = self.standardizer.inverse_transform(
                    preds_norm, batch.target_type
                )

                all_preds.append(preds_orig.cpu())
                all_targets.append(batch.y.cpu())
                # batch.target_type is tensor of indices, convert to type strings for metric calc
                type_indices = batch.target_type.cpu().numpy()
                type_strs = [Config.COUPLING_TYPES[i] for i in type_indices]
                all_types.extend(type_strs)

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        all_types = np.array(all_types)

        # Compute Metric
        score = compute_log_mae(all_preds, all_targets, all_types)
        return score

    def fit(self):
        print(f"Starting training for {Config.EPOCHS} epochs...")
        best_score = float("inf")
        patience_counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            t0 = time.time()

            # Train
            train_loss, train_coup, train_aux = self.train_epoch(epoch)

            # Validate
            val_score = self.validate()

            dt = time.time() - t0

            print(
                f"Epoch {epoch:02d} | Time: {dt:.1f}s | "
                f"Train Loss: {train_loss:.6f} (Coup: {train_coup:.6f}, Aux: {train_aux:.6f}) | "
                f"Val LogMAE: {val_score:.9f}"
            )

            # Checkpoint & Early Stopping
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  >>> New Best Model Saved! (Score: {best_score:.9f})")
            else:
                patience_counter += 1
                print(f"  >>> Patience: {patience_counter}/{Config.PATIENCE}")
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    def generate_submission(self):
        print("\nGenerating Submission...")
        # Load Best Model
        if os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            print("Warning: No saved model found. Using current model state.")

        self.model.eval()

        # Load Test Data
        test_dataset = MolecularGraphDataset(mode="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Load Test Metadata for IDs
        df_test = pd.read_csv(Config.TEST_METADATA)
        ids = df_test["id"].values

        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)

                # Forward
                preds_norm, _, _ = self.model(batch)

                # Inverse Transform
                preds_orig = self.standardizer.inverse_transform(
                    preds_norm, batch.target_type
                )

                all_preds.append(preds_orig.cpu().numpy())

        final_preds = np.concatenate(all_preds)

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id": ids, "scalar_coupling_constant": final_preds})

        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(df_sub.head())


def run_training():
    trainer = Trainer()
    trainer.fit()
    trainer.generate_submission()
