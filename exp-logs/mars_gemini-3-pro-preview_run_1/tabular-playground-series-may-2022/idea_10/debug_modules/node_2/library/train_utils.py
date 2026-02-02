import os
import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


class Trainer:
    """
    Manages the training, validation, and prediction for the SS-DeGUT model.
    """

    def __init__(self, model, train_loader, val_loader, test_loader, config: Config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.DEVICE

        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
        )

        # Scheduler (OneCycleLR)
        # Total steps = epochs * steps_per_epoch
        self.total_steps = len(train_loader) * config.EPOCHS
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LR,
            total_steps=self.total_steps,
            pct_start=config.PCT_START,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0,
        )

        self.best_auc = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training (Semi-Supervised Denoising).
        """
        self.model.train()
        running_loss = 0.0
        running_metrics = {}
        n_batches = 0

        start_time = time.time()

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            # compute_loss handles moving data to device and masking internally
            loss, metrics = self.model.compute_loss(batch, self.device)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()
            for k, v in metrics.items():
                running_metrics[k] = running_metrics.get(k, 0.0) + v

            n_batches += 1

        avg_loss = running_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in running_metrics.items()}
        elapsed = time.time() - start_time

        return avg_loss, avg_metrics, elapsed

    def validate(self):
        """
        Runs validation on the labeled validation set.
        No masking is applied. Metric: ROC AUC.
        """
        self.model.eval()
        preds = []
        targets = []
        running_loss = 0.0
        n_batches = 0

        criterion = torch.nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for batch in self.val_loader:
                x_num = batch["x_num"].to(self.device)
                x_seq = batch["x_seq"].to(self.device)
                y = batch["target"].to(self.device).float().unsqueeze(-1)

                # Forward pass without masking (inference mode)
                outputs = self.model(x_num, x_seq, mask_ratio=0.0)
                logits = outputs["logits"]

                loss = criterion(logits, y)
                running_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs)
                targets.extend(y.cpu().numpy())

                n_batches += 1

        avg_loss = running_loss / n_batches
        preds = np.array(preds).flatten()
        targets = np.array(targets).flatten()

        try:
            auc = roc_auc_score(targets, preds)
        except ValueError:
            auc = 0.5  # Fallback if only one class present in batch (unlikely)

        return avg_loss, auc

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")
        print(f"Total Epochs: {self.config.EPOCHS}")
        print(f"Batch Size: {self.config.BATCH_SIZE}")

        for epoch in range(1, self.config.EPOCHS + 1):
            # Train
            train_loss, train_metrics, train_time = self.train_epoch(epoch)

            # Validate
            val_loss, val_auc = self.validate()

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch}/{self.config.EPOCHS} | "
                f"Time: {train_time:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Train Cls: {train_metrics.get('loss_cls', 0)} | "
                f"Train Recon Num: {train_metrics.get('loss_recon_num', 0)} | "
                f"Train Recon Seq: {train_metrics.get('loss_recon_seq', 0)} | "
                f"Val Loss: {val_loss} | "
                f"Val AUC: {val_auc}"
            )

            # Early Stopping Check
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)
                print(f"New best model saved with AUC: {self.best_auc}")
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{self.config.PATIENCE}"
                )

            if self.patience_counter >= self.config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation AUC: {self.best_auc}")

    def predict(self):
        """
        Generates predictions for the test set using the best saved model.
        Saves the submission file.
        """
        print("Loading best model for prediction...")
        if not os.path.exists(self.config.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {self.config.MODEL_PATH}")

        self.model.load_state_dict(
            torch.load(self.config.MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        all_preds = []
        all_ids = []

        # We need IDs for submission.
        # The test_loader yields batches from ManufacturingDataset.
        # ManufacturingDataset returns x_num, x_seq, target.
        # We need to access the IDs separately or assume order is preserved.
        # Based on data_utils.py, test_loader is sequential (shuffle=False).
        # We can load the IDs from the cache directly to ensure alignment.

        print("Generating predictions...")
        with torch.no_grad():
            for batch in self.test_loader:
                x_num = batch["x_num"].to(self.device)
                x_seq = batch["x_seq"].to(self.device)

                # Inference without masking
                outputs = self.model(x_num, x_seq, mask_ratio=0.0)
                logits = outputs["logits"]
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                all_preds.extend(probs)

        # Load IDs from cache
        ids_path = os.path.join(self.config.CACHE_DIR, "ids_test.npy")
        if os.path.exists(ids_path):
            test_ids = np.load(ids_path)
        else:
            # Fallback: Read from metadata if cache missing (should not happen if pipeline ran)
            df_test = pd.read_csv(self.config.TEST_META_PATH)
            test_ids = df_test["id"].values

        if len(test_ids) != len(all_preds):
            raise ValueError(
                f"Mismatch in prediction count: IDs={len(test_ids)}, Preds={len(all_preds)}"
            )

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": test_ids, "target": all_preds})

        # Save
        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")

        # Preview
        print("Submission Preview:")
        print(submission.head())
