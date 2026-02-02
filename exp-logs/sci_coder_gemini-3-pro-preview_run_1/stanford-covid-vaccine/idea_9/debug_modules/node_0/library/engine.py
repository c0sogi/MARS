import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.data import get_dataloaders
from library.model import MultiTaskRNANet


class Engine:
    """
    Engine class to handle training, validation, and inference for the
    Multi-Task Distance-Aware Residual BiGRU.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model = MultiTaskRNANet().to(self.device)

        # Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss functions
        self.mse_criterion = nn.MSELoss()
        self.ce_criterion = nn.CrossEntropyLoss()

    def compute_loss(self, outputs, batch_data, mode="train"):
        """
        Computes the composite loss: Regression MSE + Lambda * Reconstruction CE.
        """
        pred_deg = outputs["pred_degradation"]  # (B, L, 5)
        pred_rec = outputs["pred_reconstruction"]  # (B, L, 4)

        targets = batch_data.get("targets")  # (B, 68, 5) or None
        seq_input = batch_data["seq_input"]  # (B, L) with masks
        rec_labels = batch_data["reconstruction_labels"]  # (B, L) original seq

        total_loss = 0.0
        losses = {}

        # 1. Regression Loss (Only if targets exist)
        if targets is not None:
            # Slice prediction to scored length (68)
            pred_deg_scored = pred_deg[:, : Config.SCORED_LEN, :]

            # Move targets to device
            targets = targets.to(self.device)

            reg_loss = self.mse_criterion(pred_deg_scored, targets)
            total_loss += reg_loss
            losses["reg_loss"] = reg_loss.item()

        # 2. Reconstruction Loss (Only on masked tokens)
        if mode == "train":
            # Identify masked positions: where input is MASK_TOKEN_ID
            mask_indices = seq_input == Config.MASK_TOKEN_ID

            if mask_indices.sum() > 0:
                # Flatten predictions and labels for masked positions
                # pred_rec: (B, L, 4) -> select masked -> (N_masked, 4)
                masked_logits = pred_rec[mask_indices]

                # rec_labels: (B, L) -> select masked -> (N_masked,)
                masked_labels = rec_labels.to(self.device)[mask_indices]

                rec_loss = self.ce_criterion(masked_logits, masked_labels)
                total_loss += Config.LAMBDA_AUX * rec_loss
                losses["rec_loss"] = rec_loss.item()
            else:
                losses["rec_loss"] = 0.0

        losses["total"] = total_loss
        return total_loss, losses

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        running_reg_loss = 0.0
        running_rec_loss = 0.0

        for batch in dataloader:
            # Move inputs to device
            seq_input = batch["seq_input"].to(self.device)
            loop_input = batch["loop_input"].to(self.device)
            dist_input = batch["dist_input"].to(self.device)

            # Forward
            outputs = self.model(seq_input, loop_input, dist_input)

            # Compute Loss
            loss, loss_dict = self.compute_loss(outputs, batch, mode="train")

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track
            running_loss += loss.item()
            running_reg_loss += loss_dict.get("reg_loss", 0.0)
            running_rec_loss += loss_dict.get("rec_loss", 0.0)

        avg_loss = running_loss / len(dataloader)
        avg_reg = running_reg_loss / len(dataloader)
        avg_rec = running_rec_loss / len(dataloader)

        return avg_loss, avg_reg, avg_rec

    def validate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                seq_input = batch["seq_input"].to(self.device)
                loop_input = batch["loop_input"].to(self.device)
                dist_input = batch["dist_input"].to(self.device)
                targets = batch[
                    "targets"
                ]  # Keep on CPU for metric calc usually, but let's see

                outputs = self.model(seq_input, loop_input, dist_input)

                # Extract degradation predictions
                pred_deg = outputs["pred_degradation"]  # (B, L, 5)

                # Slice to scored length
                pred_deg_scored = pred_deg[:, : Config.SCORED_LEN, :]

                all_preds.append(pred_deg_scored.cpu().numpy())
                all_targets.append(targets.numpy())

        # Concatenate
        all_preds = np.vstack(all_preds)  # (N_total, 68, 5)
        all_targets = np.vstack(all_targets)  # (N_total, 68, 5)

        # Calculate MCRMSE
        score = mcrmse_loss(all_targets, all_preds, num_scored=Config.SCORED_LEN)
        return score

    def run_training(self):
        print(f"Starting training on device: {self.device}")
        set_seed(Config.SEED)
        Config.create_dirs()

        # Load Data
        train_loader, val_loader, _ = get_dataloaders(
            load_cached_data=True, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS
        )

        best_mcrmse = float("inf")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            # Train
            train_loss, train_reg, train_rec = self.train_one_epoch(train_loader)

            # Validate
            val_mcrmse = self.validate(val_loader)

            # Step Scheduler
            scheduler.step()

            print(f"Epoch {epoch+1}/{Config.EPOCHS}")
            print(
                f"  Train Loss: {train_loss:.6f} (Reg: {train_reg:.6f}, Rec: {train_rec:.6f})"
            )
            print(f"  Val MCRMSE: {val_mcrmse}")  # Full precision

            # Early Stopping Check
            if val_mcrmse < best_mcrmse:
                best_mcrmse = val_mcrmse
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"  New best model saved! Score: {best_mcrmse}")
            else:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val MCRMSE: {best_mcrmse}")

    def generate_submission(self):
        print("Generating submission...")
        set_seed(Config.SEED)

        # Load Data
        _, _, test_loader = get_dataloaders(
            load_cached_data=True, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
        )

        # Load Best Model
        if not os.path.exists(Config.BEST_MODEL_PATH):
            print("No model found. Cannot generate submission.")
            return

        self.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        ids_list = []
        preds_list = []

        with torch.no_grad():
            for batch in test_loader:
                seq_input = batch["seq_input"].to(self.device)
                loop_input = batch["loop_input"].to(self.device)
                dist_input = batch["dist_input"].to(self.device)
                batch_ids = batch["id"]

                outputs = self.model(seq_input, loop_input, dist_input)
                pred_deg = outputs["pred_degradation"].cpu().numpy()  # (B, 107, 5)

                ids_list.extend(batch_ids)
                preds_list.append(pred_deg)

        # Concatenate all predictions: (N_test, 107, 5)
        all_preds = np.vstack(preds_list)

        # Format for submission
        # We need to flatten: id_seqpos, val1, val2, val3, val4, val5
        submission_data = []
        target_cols = (
            Config.TARGET_COLS
        )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        for i, sample_id in enumerate(ids_list):
            sample_preds = all_preds[i]  # (107, 5)
            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_preds[seqpos].tolist()
                submission_data.append([row_id] + row_values)

        columns = ["id_seqpos"] + target_cols
        df_sub = pd.DataFrame(submission_data, columns=columns)

        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def train_and_predict():
    """
    Wrapper function to run the full pipeline.
    """
    engine = Engine()
    engine.run_training()
    engine.generate_submission()
