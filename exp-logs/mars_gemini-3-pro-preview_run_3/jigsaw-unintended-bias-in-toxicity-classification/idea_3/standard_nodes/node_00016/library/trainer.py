import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.metrics import compute_final_score


class EarlyStopping:
    """
    Early stopping to stop the training when the score does not improve after
    certain epochs. Saves the best model state using deepcopy.
    """

    def __init__(self, patience=Config.PATIENCE, mode="max", delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
        elif self.mode == "max":
            if score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_state = copy.deepcopy(model.state_dict())
                self.counter = 0
        elif self.mode == "min":
            if score > self.best_score - self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_state = copy.deepcopy(model.state_dict())
                self.counter = 0


def loss_fn(outputs, targets, aux_targets, sample_weights):
    """
    Computes the combined loss:
    1. Weighted BCE for the main target (using sample_weights).
    2. Standard BCE for auxiliary targets.
    """
    # outputs shape: (batch_size, 7)
    # Index 0: Main Target
    # Index 1-6: Aux Targets

    main_logits = outputs[:, 0]
    aux_logits = outputs[:, 1:]

    # Main Task Loss with Sample Weighting
    # reduction='none' computes loss per element, allowing us to multiply by weights
    bce_main = nn.BCEWithLogitsLoss(reduction="none")(main_logits, targets)
    weighted_main_loss = (bce_main * sample_weights).mean()

    # Auxiliary Task Loss
    # Standard BCE averaged over batch
    bce_aux = nn.BCEWithLogitsLoss()(aux_logits, aux_targets)

    # Total Loss
    total_loss = weighted_main_loss + (Config.AUX_LOSS_WEIGHT * bce_aux)

    return total_loss


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)
        aux_targets = batch["aux_targets"].to(device)
        sample_weights = batch["sample_weight"].to(device)

        batch_size = input_ids.size(0)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)

        loss = loss_fn(outputs, targets, aux_targets, sample_weights)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, dataloader, device):
    """
    Validation loop for one epoch. Returns loss and predictions.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)
            aux_targets = batch["aux_targets"].to(device)
            sample_weights = batch["sample_weight"].to(device)

            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask)

            loss = loss_fn(outputs, targets, aux_targets, sample_weights)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Extract main target probabilities for metrics
            main_logits = outputs[:, 0]
            probs = torch.sigmoid(main_logits).cpu().numpy()
            preds.append(probs)

    epoch_loss = running_loss / dataset_size
    predictions = np.concatenate(preds)

    return epoch_loss, predictions


class Trainer:
    """
    Main class to handle training, validation, and inference.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        val_df,
        device=Config.DEVICE,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.val_df = val_df
        self.device = device
        self.early_stopping = EarlyStopping(patience=Config.PATIENCE, mode="max")

    def fit(self, epochs=Config.EPOCHS):
        self.model.to(self.device)
        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            start_time = time.time()

            # --- Train ---
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.scheduler,
                self.device,
            )

            # --- Validate ---
            val_loss, val_preds = valid_one_epoch(
                self.model, self.val_loader, self.device
            )

            # --- Metrics ---
            # Compute the competition metric
            final_score, metrics_summary = compute_final_score(self.val_df, val_preds)

            elapsed = time.time() - start_time

            # Print metrics (Full precision as requested)
            print(f"Epoch {epoch + 1}/{epochs} | Time: {elapsed:.2f}s")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Validation Score: {final_score}")
            print(f"Overall AUC: {metrics_summary['overall_auc']}")
            print(f"Subgroup Mean AUC: {metrics_summary['subgroup_mean']}")
            print(f"BPSN Mean AUC: {metrics_summary['bpsn_mean']}")
            print(f"BNSP Mean AUC: {metrics_summary['bnsp_mean']}")

            # --- Early Stopping ---
            self.early_stopping(final_score, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # Restore best model
        if self.early_stopping.best_state is not None:
            self.model.load_state_dict(self.early_stopping.best_state)
            print("Restored best model state.")

        # Save best model
        save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"Best model saved to {save_path}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        """
        self.model.eval()
        self.model.to(self.device)

        all_ids = []
        all_preds = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                ids = batch["id"]

                outputs = self.model(input_ids, attention_mask)

                # Main target probabilities
                main_logits = outputs[:, 0]
                probs = torch.sigmoid(main_logits).cpu().numpy()

                all_preds.append(probs)

                # Handle IDs (tensor or list)
                if isinstance(ids, torch.Tensor):
                    all_ids.extend(ids.cpu().numpy())
                else:
                    all_ids.extend(ids)

        predictions = np.concatenate(all_preds)
        return all_ids, predictions
