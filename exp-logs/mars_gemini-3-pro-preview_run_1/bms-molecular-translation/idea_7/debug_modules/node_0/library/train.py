import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import nltk

from library.config import Config
from library.data import get_dataloaders
from library.model import AMViT
from library.utils import seed_everything


class Trainer:
    def __init__(self, debug=False):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)

        # DataLoaders
        self.train_loader, self.val_loader, self.test_loader, self.tokenizer = (
            get_dataloaders(debug=self.debug)
        )

        # Model
        vocab_size = len(self.tokenizer)
        pad_idx = self.tokenizer.token2id[self.tokenizer.pad_token]
        self.model = AMViT(vocab_size=vocab_size, pad_idx=pad_idx).to(self.device)

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (OneCycleLR)
        # We need to know the number of steps per epoch
        self.steps_per_epoch = len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            epochs=Config.NUM_EPOCHS,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=0.1,  # Warmup for first 10%
            div_factor=25,
            final_div_factor=1000,
        )

        # Losses
        self.criterion_text = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.criterion_attr = nn.MSELoss()

        # Metrics & Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_text_loss = 0.0
        running_attr_loss = 0.0

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            # text_seq contains <SOS> ... <EOS> <PAD>
            text_seq = batch["text_seq"].to(self.device)
            attributes = batch["attributes"].to(self.device)

            # Teacher Forcing Inputs/Targets
            # Input: <SOS> ... LastToken (exclude last)
            # Target: FirstToken ... <EOS> (exclude first <SOS>)
            input_seq = text_seq[:, :-1]
            target_seq = text_seq[:, 1:]

            self.optimizer.zero_grad()

            # Forward
            # logits: (B, SeqLen-1, Vocab)
            # pred_attrs: (B, NumAttrs)
            logits, pred_attrs = self.model(images, input_seq)

            # Calculate Losses
            # Reshape for CrossEntropy: (N*L, C) vs (N*L)
            vocab_size = logits.size(-1)
            loss_text = self.criterion_text(
                logits.reshape(-1, vocab_size), target_seq.reshape(-1)
            )

            loss_attr = self.criterion_attr(pred_attrs, attributes)

            # Combined Loss
            loss = loss_text + Config.ATTR_LOSS_WEIGHT * loss_attr

            # Backward
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()
            self.scheduler.step()

            # Logging
            running_loss += loss.item()
            running_text_loss += loss_text.item()
            running_attr_loss += loss_attr.item()

        avg_loss = running_loss / len(self.train_loader)
        avg_text = running_text_loss / len(self.train_loader)
        avg_attr = running_attr_loss / len(self.train_loader)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {avg_loss:.6f} (Text: {avg_text:.6f}, Attr: {avg_attr:.6f}) | Time: {elapsed:.1f}s"
        )

        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0

        # For Levenshtein metric, we only check a subset to save time
        lev_distances = []
        check_metric_batches = 5  # Check first 5 batches for metric

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                images = batch["image"].to(self.device)
                text_seq = batch["text_seq"].to(self.device)
                attributes = batch["attributes"].to(self.device)
                original_texts = batch["original_text"]

                # Teacher Forcing Loss Calculation
                input_seq = text_seq[:, :-1]
                target_seq = text_seq[:, 1:]

                logits, pred_attrs = self.model(images, input_seq)

                vocab_size = logits.size(-1)
                loss_text = self.criterion_text(
                    logits.reshape(-1, vocab_size), target_seq.reshape(-1)
                )
                loss_attr = self.criterion_attr(pred_attrs, attributes)
                loss = loss_text + Config.ATTR_LOSS_WEIGHT * loss_attr

                running_loss += loss.item()

                # Metric Calculation (Autoregressive Generation)
                if batch_idx < check_metric_batches:
                    # Generate predictions
                    # sos_idx and eos_idx needed
                    sos_id = self.tokenizer.token2id[self.tokenizer.sos_token]
                    eos_id = self.tokenizer.token2id[self.tokenizer.eos_token]

                    generated_ids = self.model.generate(
                        images, max_len=Config.MAX_LEN, sos_idx=sos_id, eos_idx=eos_id
                    )

                    # Decode and compare
                    for i in range(len(images)):
                        pred_str = self.tokenizer.decode(generated_ids[i].cpu().numpy())
                        true_str = original_texts[i]
                        dist = nltk.edit_distance(pred_str, true_str)
                        lev_distances.append(dist)

        avg_loss = running_loss / len(self.val_loader)
        mean_lev = np.mean(lev_distances) if lev_distances else 0.0

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Val Loss: {avg_loss:.6f} | Val Levenshtein (Subset): {mean_lev:.4f}"
        )

        return avg_loss

    def fit(self):
        print(f"Starting training on device: {self.device}")

        for epoch in range(Config.NUM_EPOCHS):
            _ = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            # Early Stopping & Checkpointing
            if val_loss < self.best_val_loss:
                print(
                    f"Validation loss improved from {self.best_val_loss:.6f} to {val_loss:.6f}. Saving model..."
                )
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(
                    f"No improvement in validation loss. Patience: {self.patience_counter}/{Config.PATIENCE}"
                )

            if self.patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print("Training complete.")

    def predict(self):
        print("Starting prediction on test set...")

        # Load best model
        if not os.path.exists(Config.MODEL_PATH):
            print("No trained model found. Skipping prediction.")
            return

        self.model.load_state_dict(
            torch.load(Config.MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        image_ids = []
        predictions = []

        sos_id = self.tokenizer.token2id[self.tokenizer.sos_token]
        eos_id = self.tokenizer.token2id[self.tokenizer.eos_token]

        with torch.no_grad():
            for batch in self.test_loader:
                images = batch["image"].to(self.device)
                ids = batch["image_id"]

                # Generate
                generated_ids = self.model.generate(
                    images, max_len=Config.MAX_LEN, sos_idx=sos_id, eos_idx=eos_id
                )

                # Decode
                for i in range(len(images)):
                    pred_str = self.tokenizer.decode(generated_ids[i].cpu().numpy())
                    predictions.append(pred_str)
                    image_ids.append(ids[i])

        # Create submission dataframe
        df_sub = pd.DataFrame({"image_id": image_ids, "InChI": predictions})

        # Save
        df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
        print("First 5 predictions:")
        print(df_sub.head())


def main(debug=False):
    trainer = Trainer(debug=debug)
    trainer.fit()
    trainer.predict()


if __name__ == "__main__":
    # This block is for local testing only; the function 'main' is exposed for import.
    pass
