import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter

from library.config import Config
from library.utils import compute_class_weights, set_seed


class TaggerTrainer:
    """
    Trainer for the Prior-Informed Bi-LSTM Tagger.
    Handles class imbalance using weighted CrossEntropyLoss.
    """

    def __init__(self, model, train_dataset, val_dataset):
        self.model = model.to(Config.DEVICE)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = Config.DEVICE

        # Compute Class Weights if enabled
        weight_tensor = None
        if Config.USE_CLASS_WEIGHTS:
            print("Computing class weights for Tagger...")
            # Extract all labels from the dataset to count frequencies
            # dataset.data is a list of dicts, each has 'labels' which is a numpy array
            all_labels = []
            for item in train_dataset.data:
                # Filter out padding (0) from counting if desired,
                # but compute_class_weights handles general counts.
                # We usually care about non-pad classes.
                labels = item["labels"]
                mask = item["mask"]
                valid_labels = labels[mask]
                all_labels.extend(valid_labels.tolist())

            counts = Counter(all_labels)
            weights_dict = compute_class_weights(counts)

            # Convert to tensor, index by class ID
            # Size should be num_classes.
            # model.num_classes includes PAD at 0.
            num_classes = model.num_classes
            weights = np.ones(num_classes, dtype=np.float32)
            for cls_id, weight in weights_dict.items():
                if cls_id < num_classes:
                    weights[cls_id] = weight

            # Set PAD weight to 0 to ignore it in loss, or 1 if using ignore_index
            # Usually we use ignore_index=0 in CrossEntropyLoss
            weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=0)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.TAGGER_LR,
            weight_decay=Config.TAGGER_WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1, verbose=True
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total_tokens = 0

        for batch in dataloader:
            # Move batch to device
            word_ids = batch["word_ids"].to(self.device)
            bpe_ids = batch["bpe_ids"].to(self.device)
            char_ids = batch["char_ids"].to(self.device)
            regex_features = batch["regex_features"].to(self.device)
            prior_features = batch["prior_features"].to(self.device)
            labels = batch["labels"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(
                word_ids, bpe_ids, char_ids, regex_features, prior_features
            )
            # logits: (batch, seq, num_classes)
            # labels: (batch, seq)

            # Flatten for loss
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            loss = self.criterion(logits_flat, labels_flat)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy on valid tokens only
            predictions = torch.argmax(logits, dim=-1)
            # Apply mask
            valid_preds = predictions[mask]
            valid_labels = labels[mask]

            correct += (valid_preds == valid_labels).sum().item()
            total_tokens += valid_labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total_tokens if total_tokens > 0 else 0.0
        return avg_loss, accuracy

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                word_ids = batch["word_ids"].to(self.device)
                bpe_ids = batch["bpe_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                regex_features = batch["regex_features"].to(self.device)
                prior_features = batch["prior_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                mask = batch["mask"].to(self.device)

                logits = self.model(
                    word_ids, bpe_ids, char_ids, regex_features, prior_features
                )

                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)

                loss = self.criterion(logits_flat, labels_flat)
                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                valid_preds = predictions[mask]
                valid_labels = labels[mask]

                correct += (valid_preds == valid_labels).sum().item()
                total_tokens += valid_labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total_tokens if total_tokens > 0 else 0.0
        return avg_loss, accuracy

    def train(self):
        print(f"Starting Tagger Training on {self.device}...")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.TAGGER_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.TAGGER_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.TAGGER_EPOCHS):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.scheduler.step(val_loss)

            duration = time.time() - start_time
            print(f"Epoch {epoch+1}/{Config.TAGGER_EPOCHS} | Time: {duration}s")
            print(f"Train Loss: {train_loss} | Train Acc: {train_acc}")
            print(f"Val Loss:   {val_loss} | Val Acc:   {val_acc}")

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.TAGGER_MODEL_PATH)
                print(f"New best model saved to {Config.TAGGER_MODEL_PATH}")
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{Config.TAGGER_PATIENCE}"
                )
                if patience_counter >= Config.TAGGER_PATIENCE:
                    print("Early stopping triggered.")
                    break

        # Load best model before returning
        self.model.load_state_dict(torch.load(Config.TAGGER_MODEL_PATH))
        print("Tagger training complete.")


class Seq2SeqTrainer:
    """
    Trainer for the Character-Level LSTM Seq2Seq Fallback Model.
    Uses Teacher Forcing and CrossEntropyLoss.
    """

    def __init__(self, model, train_dataset, val_dataset):
        self.model = model.to(Config.DEVICE)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = Config.DEVICE

        # Ignore padding index (0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.SEQ2SEQ_LR,
            weight_decay=Config.SEQ2SEQ_WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1, verbose=True
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            src_char_ids = batch["src_char_ids"].to(self.device)
            tgt_char_ids = batch["tgt_char_ids"].to(self.device)
            class_id = batch["class_id"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with Teacher Forcing
            # output shape: (batch, tgt_len - 1, vocab_size)
            output = self.model(
                src_char_ids,
                tgt_char_ids,
                class_id,
                teacher_forcing_ratio=Config.TEACHER_FORCING_RATIO,
            )

            # Targets: exclude SOS token (index 0 in sequence usually, but depends on padding)
            # In data_processing, tgt_ids = [SOS] + chars + [EOS] + [PAD]...
            # We predict from SOS -> char1, char1 -> char2 ...
            # So targets are tgt_char_ids[:, 1:]
            # Output length is tgt_len - 1

            targets = tgt_char_ids[:, 1:]

            # Flatten for loss
            output_flat = output.reshape(-1, output.shape[-1])
            targets_flat = targets.reshape(-1)

            loss = self.criterion(output_flat, targets_flat)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                src_char_ids = batch["src_char_ids"].to(self.device)
                tgt_char_ids = batch["tgt_char_ids"].to(self.device)
                class_id = batch["class_id"].to(self.device)

                # Disable teacher forcing for validation (ratio = 0)
                output = self.model(
                    src_char_ids, tgt_char_ids, class_id, teacher_forcing_ratio=0.0
                )

                targets = tgt_char_ids[:, 1:]

                output_flat = output.reshape(-1, output.shape[-1])
                targets_flat = targets.reshape(-1)

                loss = self.criterion(output_flat, targets_flat)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self):
        print(f"Starting Seq2Seq Training on {self.device}...")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.SEQ2SEQ_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.SEQ2SEQ_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.SEQ2SEQ_EPOCHS):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.scheduler.step(val_loss)

            duration = time.time() - start_time
            print(f"Epoch {epoch+1}/{Config.SEQ2SEQ_EPOCHS} | Time: {duration}s")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss:   {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.SEQ2SEQ_MODEL_PATH)
                print(f"New best model saved to {Config.SEQ2SEQ_MODEL_PATH}")
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{Config.SEQ2SEQ_PATIENCE}"
                )
                if patience_counter >= Config.SEQ2SEQ_PATIENCE:
                    print("Early stopping triggered.")
                    break

        self.model.load_state_dict(torch.load(Config.SEQ2SEQ_MODEL_PATH))
        print("Seq2Seq training complete.")
