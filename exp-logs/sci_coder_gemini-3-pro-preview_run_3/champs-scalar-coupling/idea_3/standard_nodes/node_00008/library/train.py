import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import calculate_log_mae, TypeSpecificStandardizer
from library.data import ChampsDataset, get_collate_fn
from library.model import PhysicsAwareNet


class Trainer:
    def __init__(self):
        self.device = Config.get_device()
        Config.set_seed(Config.SEED)

        # Initialize Standardizer
        self.standardizer = TypeSpecificStandardizer(device=self.device)

        # Placeholders
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def setup_data(self, load_cached_data=True):
        print("Initializing Datasets...")
        # Load Train and Val datasets
        self.train_dataset = ChampsDataset(
            Config.TRAIN_META_PATH, mode="train", load_cached_data=load_cached_data
        )
        self.val_dataset = ChampsDataset(
            Config.VAL_META_PATH, mode="val", load_cached_data=load_cached_data
        )

        # Fit standardizer on training data statistics
        print("Fitting Standardizer...")
        self.standardizer.fit(
            self.train_dataset.couplings_df, load_cached_data=load_cached_data
        )

        # Create DataLoaders
        collate = get_collate_fn()
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate,
            num_workers=4,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate,
            num_workers=4,
            pin_memory=True,
        )

    def setup_model(self):
        print("Initializing Model...")
        self.model = PhysicsAwareNet().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.MAX_EPOCHS
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_aux_loss = 0.0
        count = 0

        for batch in self.train_loader:
            # Move batch to device
            batch_x = batch["x"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            edge_attr = batch["edge_attr"].to(self.device)
            triplets = batch["triplets"].to(self.device)
            triplet_attr = batch["triplet_attr"].to(self.device)

            coupling_index = batch["coupling_index"].to(self.device)
            coupling_type = batch["coupling_type"].to(self.device)
            coupling_value = batch["coupling_value"].to(self.device)

            aux_targets = batch["aux"].to(self.device)  # [N, 10]

            self.optimizer.zero_grad()

            # Forward Pass
            pred_coupling, pred_shield, pred_charge = self.model(
                x=batch_x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                triplets=triplets,
                triplet_attr=triplet_attr,
                coupling_index=coupling_index,
                coupling_type=coupling_type,
            )

            # --- Loss Calculation ---

            # 1. Main Task: Standardized L1 Loss
            # Standardize targets
            target_std = self.standardizer.transform(coupling_value, coupling_type)
            loss_coupling = nn.functional.l1_loss(pred_coupling, target_std)

            # 2. Auxiliary Tasks: Physics Regularization
            # Shielding (First 9 columns)
            loss_shield = nn.functional.l1_loss(pred_shield, aux_targets[:, :9])

            # Charge (Last column) - unsqueeze to match pred shape [N, 1]
            loss_charge = nn.functional.l1_loss(
                pred_charge, aux_targets[:, 9].unsqueeze(-1)
            )

            # Weighted Sum
            loss = (
                loss_coupling
                + Config.LAMBDA_SHIELDING * loss_shield
                + Config.LAMBDA_CHARGE * loss_charge
            )

            # Backward
            loss.backward()
            self.optimizer.step()

            # Logging
            bs = coupling_value.size(0)
            total_loss += loss.item() * bs
            total_main_loss += loss_coupling.item() * bs
            total_aux_loss += (loss_shield.item() + loss_charge.item()) * bs
            count += bs

        avg_loss = total_loss / count
        avg_main = total_main_loss / count
        avg_aux = total_aux_loss / count

        return avg_loss, avg_main, avg_aux

    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []
        all_types = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch_x = batch["x"].to(self.device)
                edge_index = batch["edge_index"].to(self.device)
                edge_attr = batch["edge_attr"].to(self.device)
                triplets = batch["triplets"].to(self.device)
                triplet_attr = batch["triplet_attr"].to(self.device)

                coupling_index = batch["coupling_index"].to(self.device)
                coupling_type = batch["coupling_type"].to(self.device)
                coupling_value = batch["coupling_value"].to(self.device)

                # Forward Pass (Aux heads ignored for validation metric)
                pred_std, _, _ = self.model(
                    x=batch_x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    triplets=triplets,
                    triplet_attr=triplet_attr,
                    coupling_index=coupling_index,
                    coupling_type=coupling_type,
                )

                # Inverse Transform to original scale
                pred_orig = self.standardizer.inverse_transform(pred_std, coupling_type)

                all_preds.append(pred_orig.cpu())
                all_targets.append(coupling_value.cpu())
                all_types.append(coupling_type.cpu())

        # Concatenate
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()
        types = torch.cat(all_types).numpy()

        # Calculate Metric
        log_mae = calculate_log_mae(y_true, y_pred, types)
        return log_mae

    def run_training(self):
        self.setup_data()
        self.setup_model()

        best_score = float("inf")
        patience_counter = 0

        print(f"Starting training on device: {self.device}")

        for epoch in range(1, Config.MAX_EPOCHS + 1):
            train_loss, train_main, train_aux = self.train_epoch(epoch)
            val_score = self.validate()
            self.scheduler.step()

            print(
                f"Epoch {epoch}/{Config.MAX_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} (Main: {train_main:.6f}, Aux: {train_aux:.6f}) | "
                f"Val LogMAE: {val_score}"
            )

            # Checkpoint
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print("  -> New Best Model Saved!")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Val LogMAE: {best_score}")

    def generate_submission(self):
        print("Generating Submission...")

        # Load Test Data
        test_dataset = ChampsDataset(
            Config.TEST_META_PATH, mode="test", load_cached_data=True
        )
        collate = get_collate_fn()
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate,
            num_workers=4,
            pin_memory=True,
        )

        # Load Best Model
        self.setup_model()  # Re-init architecture
        self.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        # Ensure standardizer is ready (stats loaded)
        self.standardizer._ensure_loaded()

        ids_list = []
        preds_list = []

        with torch.no_grad():
            for batch in test_loader:
                batch_x = batch["x"].to(self.device)
                edge_index = batch["edge_index"].to(self.device)
                edge_attr = batch["edge_attr"].to(self.device)
                triplets = batch["triplets"].to(self.device)
                triplet_attr = batch["triplet_attr"].to(self.device)

                coupling_index = batch["coupling_index"].to(self.device)
                coupling_type = batch["coupling_type"].to(self.device)
                ids = batch["id"]

                # Forward
                pred_std, _, _ = self.model(
                    x=batch_x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    triplets=triplets,
                    triplet_attr=triplet_attr,
                    coupling_index=coupling_index,
                    coupling_type=coupling_type,
                )

                # Inverse Transform
                pred_orig = self.standardizer.inverse_transform(pred_std, coupling_type)

                ids_list.append(ids.cpu().numpy())
                preds_list.append(pred_orig.cpu().numpy())

        # Create DataFrame
        all_ids = np.concatenate(ids_list)
        all_preds = np.concatenate(preds_list)

        df_sub = pd.DataFrame({"id": all_ids, "scalar_coupling_constant": all_preds})

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    trainer = Trainer()
    trainer.run_training()
    trainer.generate_submission()


if __name__ == "__main__":
    main()
