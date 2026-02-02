import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from library.config import Config
from library.utils import seed_everything
from library.loss import AggressiveMultiTaskLoss
from library.metrics import calculate_jigsaw_metrics
from library.model import MultiTaskRoberta


class Engine:
    """
    Main engine class to handle training, evaluation, and inference.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.model = MultiTaskRoberta().to(self.device)
        self.loss_fn = AggressiveMultiTaskLoss().to(self.device)
        self.best_score = -float("inf")

    def trim_batch(self, input_ids, attention_mask):
        """
        Device-side trimming: Slices the tensors to the maximum valid sequence length
        in the current batch to avoid computing on padding.
        """
        # attention_mask is 1 for valid tokens, 0 for padding.
        # Find the maximum index where mask is 1.
        # sum(dim=1) gives the length of valid tokens for each sample.
        max_len = attention_mask.sum(dim=1).max().item()

        # Ensure at least 1 token to prevent errors if batch is empty (unlikely)
        max_len = max(int(max_len), 1)

        return input_ids[:, :max_len], attention_mask[:, :max_len]

    def train_one_epoch(self, dataloader, optimizer, scheduler, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["target"].to(self.device)
            identities = batch["identities"].to(self.device)

            # Optimization: Trim padding before forward pass
            input_ids, attention_mask = self.trim_batch(input_ids, attention_mask)

            optimizer.zero_grad()

            # Forward pass
            toxicity_logits, identity_logits = self.model(input_ids, attention_mask)

            # Calculate Loss
            loss = self.loss_fn(toxicity_logits, identity_logits, targets, identities)

            # Backward pass
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_loss:.6f}")
        return avg_loss

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set and calculates bias metrics.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        all_identities = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Keep targets/identities on CPU for metric calculation
                targets = batch["target"].numpy()
                identities = batch["identities"].numpy()

                # Optimization: Trim padding
                input_ids, attention_mask = self.trim_batch(input_ids, attention_mask)

                # Forward pass
                toxicity_logits, _ = self.model(input_ids, attention_mask)

                # Apply sigmoid to get probabilities
                preds = torch.sigmoid(toxicity_logits).cpu().numpy()

                all_preds.append(preds)
                all_targets.append(targets)
                all_identities.append(identities)

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_identities = np.concatenate(all_identities, axis=0)

        # Reconstruct DataFrame for metrics
        # calculate_jigsaw_metrics expects a DataFrame with identity columns and target
        val_df = pd.DataFrame(all_identities, columns=Config.IDENTITY_COLUMNS)
        val_df[Config.TARGET_COL] = all_targets

        # Calculate metrics
        metrics = calculate_jigsaw_metrics(val_df, all_preds)
        return metrics

    def predict(self, dataloader):
        """
        Generates predictions for the test set.
        """
        self.model.eval()
        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                ids = batch["id"].numpy()

                # Optimization: Trim padding
                input_ids, attention_mask = self.trim_batch(input_ids, attention_mask)

                # Forward pass (only need toxicity logits)
                toxicity_logits, _ = self.model(input_ids, attention_mask)
                preds = torch.sigmoid(toxicity_logits).cpu().numpy().flatten()

                all_ids.append(ids)
                all_preds.append(preds)

        return np.concatenate(all_ids), np.concatenate(all_preds)

    def run_training(self, train_loader, val_loader, test_loader):
        """
        Orchestrates the training process, evaluation, and submission generation.
        """
        seed_everything(Config.SEED)

        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        num_training_steps = len(train_loader) * Config.EPOCHS
        scheduler = OneCycleLR(
            optimizer,
            max_lr=Config.LEARNING_RATE,
            total_steps=num_training_steps,
            pct_start=Config.WARMUP_RATIO,
            anneal_strategy="cos",
        )

        print(f"Starting training on {self.device} for {Config.EPOCHS} epochs.")

        for epoch in range(Config.EPOCHS):
            # 1. Train
            self.train_one_epoch(train_loader, optimizer, scheduler, epoch)

            # 2. Validate
            metrics = self.evaluate(val_loader)
            score = metrics["final_score"]

            print(f"Epoch {epoch + 1} | Validation Score: {score}")
            print(f"  Overall AUC: {metrics['overall_auc']}")
            print(f"  Subgroup AUC: {metrics['subgroup_auc']}")
            print(f"  BPSN AUC: {metrics['bpsn_auc']}")
            print(f"  BNSP AUC: {metrics['bnsp_auc']}")

            # 3. Save Best Model
            if score > self.best_score:
                print(
                    f"  New best score! ({self.best_score} -> {score}). Saving model..."
                )
                self.best_score = score
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)

        print("Training complete.")

        # 4. Generate Submission
        print(f"Loading best model from {Config.MODEL_SAVE_PATH}...")
        self.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )

        print("Generating predictions on test set...")
        ids, preds = self.predict(test_loader)

        submission_df = pd.DataFrame({"id": ids, "prediction": preds})

        # Ensure output directory exists
        submission_dir = os.path.dirname(Config.SUBMISSION_PATH)
        if submission_dir:
            os.makedirs(submission_dir, exist_ok=True)

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
