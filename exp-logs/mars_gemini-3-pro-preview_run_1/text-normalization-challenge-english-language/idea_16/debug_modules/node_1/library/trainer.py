import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import json
import time
from library.config import Config
from library.dataset import get_tagger_loader, get_seq2seq_loader
from library.models import MorphoBiLSTMTagger, CharSeq2Seq


class TaggerTrainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        print(f"TaggerTrainer initializing on device: {self.device}")

        # Data Loaders
        self.train_loader = get_tagger_loader(
            "train", batch_size=Config.BATCH_SIZE, shuffle=True
        )
        self.val_loader = get_tagger_loader(
            "val", batch_size=Config.BATCH_SIZE, shuffle=False
        )

        # Vocab Sizes
        # We need to load the vocab files to know the sizes for model initialization
        with open(os.path.join(Config.VOCAB_DIR, "vocab_words.json"), "r") as f:
            word_vocab = json.load(f)
        with open(os.path.join(Config.VOCAB_DIR, "vocab_classes.json"), "r") as f:
            class_vocab = json.load(f)
        with open(os.path.join(Config.VOCAB_DIR, "vocab_chars.json"), "r") as f:
            char_vocab = json.load(f)

        self.word_vocab_size = len(word_vocab)
        self.class_vocab_size = len(class_vocab)
        self.char_vocab_size = len(char_vocab)

        # Model
        self.model = MorphoBiLSTMTagger(
            word_vocab_size=self.word_vocab_size,
            class_vocab_size=self.class_vocab_size,
            char_vocab_size=self.char_vocab_size,
        ).to(self.device)

        # Loss with Class Weights
        weights = self._calculate_class_weights(class_vocab)
        self.criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)

        # Optimizer & Scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _calculate_class_weights(self, class_vocab):
        """
        Computes square-root smoothed class weights: sqrt(Total / Count_c).
        """
        print("Calculating class weights...")
        df = pd.read_csv(Config.TRAIN_DATA_PATH, keep_default_na=False)

        # Count classes
        class_counts = df["class"].value_counts().to_dict()
        total_samples = len(df)

        # Create weight vector ordered by class ID
        weights = torch.ones(len(class_vocab), dtype=torch.float32)

        for class_name, class_id in class_vocab.items():
            count = class_counts.get(class_name, 0)
            if count > 0:
                # Square-root smoothing
                w = np.sqrt(total_samples / count)
                weights[class_id] = w
            else:
                # Fallback for classes not in train split (unlikely but safe)
                weights[class_id] = 1.0

        return weights.to(self.device)

    def train(self):
        print("Starting Tagger Training...")

        for epoch in range(Config.NUM_EPOCHS_TAGGER):
            start_time = time.time()

            train_loss = self._train_epoch()
            val_loss, val_acc = self._validate()

            self.scheduler.step(val_loss)

            duration = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS_TAGGER} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f} | "
                f"Val Acc: {val_acc:.8f}"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                save_path = os.path.join(Config.CHECKPOINT_DIR, "tagger_best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved to {save_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.PATIENCE_TAGGER:
                    print("Early stopping triggered.")
                    break

    def _train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            word_ids = batch["word_ids"].to(self.device)
            char_features = batch["char_features"].to(self.device)
            regex_features = batch["regex_features"].to(self.device)
            lengths = batch[
                "lengths"
            ]  # CPU is fine for pack_padded_sequence usually, but model handles it
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(word_ids, char_features, regex_features, lengths)

            # Flatten for loss
            # logits: [Batch, Seq, NumClasses] -> [Batch*Seq, NumClasses]
            # labels: [Batch, Seq] -> [Batch*Seq]
            loss = self.criterion(
                logits.view(-1, self.class_vocab_size), labels.view(-1)
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.val_loader:
                word_ids = batch["word_ids"].to(self.device)
                char_features = batch["char_features"].to(self.device)
                regex_features = batch["regex_features"].to(self.device)
                lengths = batch["lengths"]
                labels = batch["labels"].to(self.device)

                logits = self.model(word_ids, char_features, regex_features, lengths)

                # Loss
                loss = self.criterion(
                    logits.view(-1, self.class_vocab_size), labels.view(-1)
                )
                total_loss += loss.item()

                # Accuracy
                # Mask out padding/ignore_index (-100)
                preds = torch.argmax(logits, dim=2)
                mask = labels != -100

                correct += (preds[mask] == labels[mask]).sum().item()
                total_tokens += mask.sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total_tokens if total_tokens > 0 else 0.0
        return avg_loss, accuracy


class Seq2SeqTrainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        print(f"Seq2SeqTrainer initializing on device: {self.device}")

        # Data Loaders
        self.train_loader = get_seq2seq_loader(
            "train", batch_size=Config.BATCH_SIZE, shuffle=True
        )
        self.val_loader = get_seq2seq_loader(
            "val", batch_size=Config.BATCH_SIZE, shuffle=False
        )

        # Vocab Sizes
        with open(os.path.join(Config.VOCAB_DIR, "vocab_seq2seq.json"), "r") as f:
            char_vocab = json.load(f)
        with open(os.path.join(Config.VOCAB_DIR, "vocab_classes.json"), "r") as f:
            class_vocab = json.load(f)

        self.char_vocab_size = len(char_vocab)
        self.num_classes = len(class_vocab)

        # Model
        self.model = CharSeq2Seq(
            char_vocab_size=self.char_vocab_size, num_classes=self.num_classes
        ).to(self.device)

        # Loss
        # Ignore padding index (0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train(self):
        print("Starting Seq2Seq Training...")

        for epoch in range(Config.NUM_EPOCHS_SEQ2SEQ):
            start_time = time.time()

            train_loss = self._train_epoch()
            val_loss = self._validate()

            duration = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS_SEQ2SEQ} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f}"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                save_path = os.path.join(
                    Config.CHECKPOINT_DIR, "seq2seq_best_model.pth"
                )
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved to {save_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.PATIENCE_SEQ2SEQ:
                    print("Early stopping triggered.")
                    break

    def _train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            src_ids = batch["src_ids"].to(self.device)
            src_lens = batch["src_lens"]
            tgt_ids = batch["tgt_ids"].to(self.device)
            class_ids = batch["class_ids"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with Teacher Forcing
            # logits: [Batch, TgtLen-1, Vocab]
            logits = self.model(src_ids, src_lens, class_ids, tgt_ids)

            # Targets: Shifted right by 1 (ignore SOS at index 0)
            # tgt_ids: [Batch, TgtLen] -> targets: [Batch, TgtLen-1]
            targets = tgt_ids[:, 1:]

            # Flatten
            loss = self.criterion(
                logits.reshape(-1, self.char_vocab_size), targets.reshape(-1)
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                src_ids = batch["src_ids"].to(self.device)
                src_lens = batch["src_lens"]
                tgt_ids = batch["tgt_ids"].to(self.device)
                class_ids = batch["class_ids"].to(self.device)

                logits = self.model(src_ids, src_lens, class_ids, tgt_ids)

                targets = tgt_ids[:, 1:]
                loss = self.criterion(
                    logits.reshape(-1, self.char_vocab_size), targets.reshape(-1)
                )

                total_loss += loss.item()

        return total_loss / len(self.val_loader)
