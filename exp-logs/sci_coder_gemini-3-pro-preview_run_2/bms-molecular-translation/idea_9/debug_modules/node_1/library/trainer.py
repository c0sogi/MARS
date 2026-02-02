import os
import time
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from library.config import Config
from library.utils import set_seed, compute_levenshtein
from library.dataset import get_dataloaders
from library.model import HybridCTCAttentionModel
from library.loss import HybridLoss


class Trainer:
    def __init__(self, debug=False):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader, self.tokenizer = (
            get_dataloaders(debug=self.debug)
        )

        # Model
        self.model = HybridCTCAttentionModel().to(self.device)

        # Loss
        self.criterion = HybridLoss().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        # Scheduler
        # Calculate total steps for OneCycleLR
        steps_per_epoch = len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LR,
            epochs=Config.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=Config.PCT_START,
            div_factor=25,
            final_div_factor=1000,
        )

        # Mixed Precision
        self.scaler = GradScaler()

        # Early Stopping
        self.best_score = float("inf")
        self.patience_counter = 0
        self.early_stopping_patience = Config.PATIENCE

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_loss_ctc = 0.0
        running_loss_attn = 0.0
        start_time = time.time()

        for i, (images, sequences, lengths) in enumerate(self.train_loader):
            images = images.to(self.device)
            sequences = sequences.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                # Forward pass
                # ctc_logits: (B, T_enc, Vocab)
                # attn_logits: (B, Seq_Len, Vocab)
                ctc_logits, attn_logits = self.model(images, sequences)

                # Calculate loss
                loss, metrics = self.criterion(
                    ctc_logits, attn_logits, sequences, lengths
                )

            # Backward pass with scaler
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            running_loss += metrics["loss_total"]
            running_loss_ctc += metrics["loss_ctc"]
            running_loss_attn += metrics["loss_attn"]

        avg_loss = running_loss / len(self.train_loader)
        avg_loss_ctc = running_loss_ctc / len(self.train_loader)
        avg_loss_attn = running_loss_attn / len(self.train_loader)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_loss:.6f} (CTC: {avg_loss_ctc:.6f}, Attn: {avg_loss_attn:.6f}) | Time: {elapsed:.0f}s"
        )
        return avg_loss

    def validate(self):
        self.model.eval()
        predictions = []
        ground_truths = []

        print("Starting validation...")
        start_time = time.time()

        with torch.no_grad():
            for images, sequences, lengths in self.val_loader:
                images = images.to(self.device)

                # Predict using greedy decoding from the Attention Head
                # This returns indices: (B, Max_Len)
                pred_indices = self.model.predict(images, max_len=Config.MAX_SEQ_LEN)

                # Convert predictions to strings
                for idx_seq in pred_indices:
                    text = self.tokenizer.sequence_to_text(idx_seq)
                    predictions.append(text)

                # Convert ground truths to strings
                for seq in sequences:
                    text = self.tokenizer.sequence_to_text(seq)
                    ground_truths.append(text)

        # Compute Metric
        lev_distance = compute_levenshtein(predictions, ground_truths)
        elapsed = time.time() - start_time

        print(f"Validation Levenshtein Distance: {lev_distance} | Time: {elapsed:.0f}s")
        return lev_distance

    def fit(self):
        print(f"Starting training on {self.device}...")

        for epoch in range(Config.EPOCHS):
            _ = self.train_epoch(epoch)
            val_score = self.validate()

            # Checkpoint and Early Stopping
            if val_score < self.best_score:
                print(
                    f"Score improved from {self.best_score} to {val_score}. Saving checkpoint..."
                )
                self.best_score = val_score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.CHECKPOINT_PATH)
            else:
                self.patience_counter += 1
                print(
                    f"Score did not improve. Patience: {self.patience_counter}/{self.early_stopping_patience}"
                )
                if self.patience_counter >= self.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation Score: {self.best_score}")

    def predict_test_set(self):
        print("Predicting test set...")

        # Load best model
        if os.path.exists(Config.CHECKPOINT_PATH):
            print(f"Loading checkpoint from {Config.CHECKPOINT_PATH}")
            self.model.load_state_dict(
                torch.load(Config.CHECKPOINT_PATH, map_location=self.device)
            )
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()
        results = []

        with torch.no_grad():
            for images, image_ids in self.test_loader:
                images = images.to(self.device)

                # Predict
                pred_indices = self.model.predict(images, max_len=Config.MAX_SEQ_LEN)

                # Decode
                for i, idx_seq in enumerate(pred_indices):
                    text = self.tokenizer.sequence_to_text(idx_seq)
                    results.append({"image_id": image_ids[i], "InChI": text})

        # Save submission
        df_sub = pd.DataFrame(results)

        # Ensure submission directory exists (handled by Config.setup, but good practice)
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {df_sub.shape}")
