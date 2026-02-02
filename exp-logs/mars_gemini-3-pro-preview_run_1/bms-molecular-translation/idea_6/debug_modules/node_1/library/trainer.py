import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import compute_levenshtein
from library.tokenizer import get_tokenizer


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, tokenizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.device = torch.device(Config.DEVICE)

        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1, verbose=True
        )

        # Loss Functions
        # Ignore padding index in CrossEntropyLoss
        self.criterion_text = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
        self.criterion_attr = nn.MSELoss()

        self.best_metric = float("inf")
        self.patience_counter = 0

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0
        running_text_loss = 0.0
        running_attr_loss = 0.0

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            attributes = batch["attributes"].to(self.device)
            seq = batch["seq"].to(self.device)

            # Teacher forcing preparation
            # Input to decoder: <SOS> ... token_n
            # Target for decoder: token_1 ... <EOS>
            decoder_input = seq[:, :-1]
            decoder_target = seq[:, 1:]

            self.optimizer.zero_grad()

            # Forward pass
            # logits shape: (B, Seq_Len-1, Vocab_Size)
            # pred_attrs shape: (B, Num_Attributes)
            logits, pred_attrs = self.model(images, decoder_input)

            # Calculate Losses
            # Flatten logits and targets for CrossEntropy
            loss_text = self.criterion_text(
                logits.reshape(-1, Config.VOCAB_SIZE), decoder_target.reshape(-1)
            )

            loss_attr = self.criterion_attr(pred_attrs, attributes)

            # Multi-task weighted loss
            loss = loss_text + (Config.ATTRIBUTE_LOSS_WEIGHT * loss_attr)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            self.optimizer.step()

            # Logging
            running_loss += loss.item()
            running_text_loss += loss_text.item()
            running_attr_loss += loss_attr.item()

        end_time = time.time()
        epoch_loss = running_loss / len(self.train_loader)
        epoch_text_loss = running_text_loss / len(self.train_loader)
        epoch_attr_loss = running_attr_loss / len(self.train_loader)

        print(
            f"Epoch {epoch_idx+1} Training: "
            f"Loss={epoch_loss:.6f} (Text={epoch_text_loss:.6f}, Attr={epoch_attr_loss:.6f}) "
            f"Time={end_time - start_time:.2f}s"
        )

        return epoch_loss

    def validate(self, epoch_idx):
        self.model.eval()
        running_loss = 0.0

        # For Levenshtein calculation
        all_preds = []
        all_truths = []

        start_time = time.time()

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                attributes = batch["attributes"].to(self.device)
                seq = batch["seq"].to(self.device)

                # 1. Calculate Validation Loss (Teacher Forcing)
                decoder_input = seq[:, :-1]
                decoder_target = seq[:, 1:]

                logits, pred_attrs = self.model(images, decoder_input)

                loss_text = self.criterion_text(
                    logits.reshape(-1, Config.VOCAB_SIZE), decoder_target.reshape(-1)
                )
                loss_attr = self.criterion_attr(pred_attrs, attributes)
                loss = loss_text + (Config.ATTRIBUTE_LOSS_WEIGHT * loss_attr)

                running_loss += loss.item()

                # 2. Calculate Levenshtein Distance (Inference)
                # We perform greedy decoding to get the actual predicted strings
                # Note: This is computationally expensive, so for very large datasets
                # one might validate on a subset or less frequently.
                # Here we do it for the full validation set as requested.
                pred_seqs = self.model.predict(images, max_len=Config.MAX_LEN)

                # Decode sequences to text
                pred_texts = [self.tokenizer.sequence_to_text(s) for s in pred_seqs]
                truth_texts = [self.tokenizer.sequence_to_text(s) for s in seq]

                all_preds.extend(pred_texts)
                all_truths.extend(truth_texts)

        epoch_loss = running_loss / len(self.val_loader)

        # Compute Metric
        lev_score = compute_levenshtein(all_preds, all_truths)

        end_time = time.time()

        print(
            f"Epoch {epoch_idx+1} Validation: "
            f"Loss={epoch_loss:.6f} "
            f"Levenshtein={lev_score} "
            f"Time={end_time - start_time:.2f}s"
        )

        return epoch_loss, lev_score

    def fit(self):
        print(f"Starting training for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_lev = self.validate(epoch)

            # Scheduler step
            self.scheduler.step(val_lev)

            # Checkpointing based on Levenshtein distance (lower is better)
            if val_lev < self.best_metric:
                print(
                    f"Metric improved from {self.best_metric} to {val_lev}. Saving model..."
                )
                self.best_metric = val_lev
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                self.patience_counter += 1
                print(
                    f"Metric did not improve. Patience: {self.patience_counter}/{Config.PATIENCE}"
                )

            if self.patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Levenshtein: {self.best_metric}")

    def predict_and_submit(self):
        print("Generating submission...")

        # Load best model
        if os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            print("Warning: Best model not found. Using current model state.")

        self.model.eval()

        results = []

        with torch.no_grad():
            for batch in self.test_loader:
                images = batch["image"].to(self.device)
                image_ids = batch["image_id"]

                # Run inference
                pred_seqs = self.model.predict(images, max_len=Config.MAX_LEN)

                # Decode
                pred_texts = [self.tokenizer.sequence_to_text(s) for s in pred_seqs]

                for img_id, text in zip(image_ids, pred_texts):
                    results.append({"image_id": img_id, "InChI": text})

        # Create DataFrame
        submission_df = pd.DataFrame(results)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Total predictions: {len(submission_df)}")
