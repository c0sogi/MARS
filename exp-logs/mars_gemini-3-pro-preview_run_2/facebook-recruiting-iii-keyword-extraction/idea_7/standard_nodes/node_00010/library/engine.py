import torch
import numpy as np
import pandas as pd
import gc
import os
from library.config import Config
from library.utils import (
    seed_everything,
    save_checkpoint,
    optimize_threshold,
    load_checkpoint,
)
from library.loss import FocalLoss
from library.model import WideDeepTextCNN
from library.dataset import get_dataloader
from library.preprocessing import Preprocessor


class Trainer:
    """
    Encapsulates the training, evaluation, and inference logic for the Wide-and-Deep model.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        seed_everything(Config.SEED)

        # Initialize Model
        self.model = WideDeepTextCNN().to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function
        self.criterion = FocalLoss()

        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler()

    def train_one_epoch(self, loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch in loader:
            # Move data to device
            wide = batch["wide"].to(self.device, non_blocking=True)
            deep = batch["deep"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)

            batch_size = wide.size(0)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with torch.cuda.amp.autocast():
                outputs = self.model({"wide": wide, "deep": deep})
                loss = self.criterion(outputs, targets)

            # Backward Pass with Scaler
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        return running_loss / dataset_size

    def evaluate(self, loader):
        """
        Evaluates the model on the validation set.
        Returns average loss, probabilities, and targets.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                wide = batch["wide"].to(self.device, non_blocking=True)
                deep = batch["deep"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)

                batch_size = wide.size(0)

                with torch.cuda.amp.autocast():
                    logits = self.model({"wide": wide, "deep": deep})
                    loss = self.criterion(logits, targets)

                # Apply Sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_loss = running_loss / dataset_size
        return avg_loss, np.vstack(all_probs), np.vstack(all_targets)

    def predict(self, loader):
        """
        Generates predictions for the test set.
        """
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                wide = batch["wide"].to(self.device, non_blocking=True)
                deep = batch["deep"].to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    logits = self.model({"wide": wide, "deep": deep})

                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def run(self):
        """
        Main execution method: Training -> Validation -> Inference.
        """
        print("Initializing Data Loaders...")
        train_loader = get_dataloader("train", shuffle=True)
        val_loader = get_dataloader("val", shuffle=False)

        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0

        print(f"Starting Training for {Config.EPOCHS} epochs...")
        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_probs, val_targets = self.evaluate(val_loader)

            print(
                f"Epoch {epoch}/{Config.EPOCHS} - Train Loss: {train_loss:.8f} - Val Loss: {val_loss:.8f}"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, Config.MODEL_SAVE_PATH
                )
                print(f"Saved Best Model at Epoch {epoch} (Loss: {best_val_loss:.8f})")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early Stopping Triggered.")
                break

            # Cleanup to save memory
            gc.collect()
            torch.cuda.empty_cache()

        # ---------------------------------------------------------
        # Inference Phase
        # ---------------------------------------------------------
        print("\nLoading Best Model for Inference...")
        load_checkpoint(Config.MODEL_SAVE_PATH, self.model, device=self.device)

        # 1. Optimize Threshold on Validation Set
        print("Optimizing Threshold on Validation Set...")
        # Re-run evaluation to ensure we have probs from the exact best model state
        _, val_probs, val_targets = self.evaluate(val_loader)
        best_threshold, best_f1 = optimize_threshold(val_targets, val_probs)
        print(f"Selected Threshold: {best_threshold} (Val F1: {best_f1})")

        # 2. Predict on Test Set
        print("Generating Test Predictions...")
        test_loader = get_dataloader("test", shuffle=False)
        test_probs = self.predict(test_loader)

        # 3. Apply Threshold
        test_preds_bin = (test_probs >= best_threshold).astype(int)

        # 4. Convert to Tag Strings
        print("Converting predictions to tags...")
        preprocessor = Preprocessor()
        # Manually load the tag encoder to ensure mappings are available
        preprocessor.tag_encoder.load(Config.TAG_ENCODER_PATH)

        pred_tags = preprocessor.inverse_transform_tags(test_preds_bin)
        test_ids = preprocessor.get_test_ids()

        # 5. Save Submission
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        df_sub = pd.DataFrame({"Id": test_ids, "Tags": pred_tags})
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission Saved.")
