import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library import utils, data, model


class Trainer:
    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, scheduler, device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_score = float("inf")

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            inputs = batch["inputs"].to(self.device)
            pair_indices = batch["pair_indices"].to(self.device)
            pair_mask = batch["pair_mask"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs, pair_indices, pair_mask)

            # Calculate loss on all 5 targets
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping (Mandatory for HC-SDBR-BiGRU)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["inputs"].to(self.device)
                pair_indices = batch["pair_indices"].to(self.device)
                pair_mask = batch["pair_mask"].to(self.device)
                targets = batch["targets"].to(self.device)

                outputs = self.model(inputs, pair_indices, pair_mask)

                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate metric using the specific slicing logic in utils
        score = utils.calculate_metric(all_preds, all_targets)
        return score

    def fit(self, epochs, patience=5):
        print(f"Starting training on device: {self.device}")

        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_score = self.validate()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
            )

            # Early Stopping and Model Checkpointing
            if val_score < self.best_score:
                self.best_score = val_score
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"New best model saved with score: {self.best_score}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break


def generate_submission(model_instance, test_loader, device):
    print("Generating submission...")

    # Load best model weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        model_instance.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=device)
        )
    else:
        print("Warning: Best model not found, using current weights.")

    model_instance.eval()
    model_instance.to(device)

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            ids = batch["id"]

            outputs = model_instance(inputs, pair_indices, pair_mask)

            # Outputs shape: (Batch, 107, 5)
            preds_list.append(outputs.cpu().numpy())
            ids_list.extend(ids)

    # Concatenate all predictions: (Total_Test_Samples, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare data for DataFrame
    submission_data = []

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # Shape (107, 5)

        for seq_pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seq_pos}"
            row_preds = sample_preds[seq_pos]

            # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            # Config.TARGET_COLS order matches the output head
            row_dict = {
                "id_seqpos": row_id,
                "reactivity": row_preds[0],
                "deg_Mg_pH10": row_preds[1],
                "deg_pH10": row_preds[2],
                "deg_Mg_50C": row_preds[3],
                "deg_50C": row_preds[4],
            }
            submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Ensure column order
    cols = ["id_seqpos"] + Config.TARGET_COLS
    submission_df = submission_df[cols]

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(load_cached_data=True, debug=False):
    # 1. Setup
    utils.seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    train_loader, val_loader, test_loader = data.get_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    # 3. Model Initialization
    net = model.HC_SDBR_BiGRU().to(device)

    # 4. Optimization
    criterion = utils.MCRMSELoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 5. Training
    trainer = Trainer(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    trainer.fit(epochs=Config.EPOCHS)

    # 6. Submission
    generate_submission(net, test_loader, device)
