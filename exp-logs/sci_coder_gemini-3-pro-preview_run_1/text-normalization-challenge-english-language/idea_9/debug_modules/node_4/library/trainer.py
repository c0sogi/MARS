import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, get_device, compute_class_weights
from library.data_loader import get_tagger_loaders, get_seq2seq_loaders
from library.model_tagger import MultiGranularityTagger
from library.model_seq2seq import TransformerFallback


class TaggerTrainer:
    """
    Trainer for the Multi-Granularity Bi-LSTM Tagger.
    """

    def __init__(self, debug=Config.DEBUG):
        self.device = get_device()
        self.debug = debug
        set_seed(Config.SEED)

        print(f"Initializing TaggerTrainer (Device: {self.device})...")

        # 1. Load Data
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.word_vocab,
            self.char_vocab,
            self.class_vocab,
            self.bpe_tokenizer,
        ) = get_tagger_loaders(debug=debug)

        # 2. Compute Class Weights
        print("Computing class weights for Tagger...")
        # Load train metadata to count classes
        if debug:
            df_train = pd.read_csv(Config.TRAIN_FILE).head(Config.MAX_DEBUG_SAMPLES)
        else:
            df_train = pd.read_csv(Config.TRAIN_FILE)

        # Map class strings to indices
        class_counts_series = df_train["class"].value_counts()
        class_counts_dict = {}
        for cls_name, count in class_counts_series.items():
            idx = self.class_vocab.stoi.get(cls_name)
            if idx is not None:
                class_counts_dict[idx] = count

        # Compute weights
        weights_tensor = compute_class_weights(
            class_counts_dict,
            len(self.class_vocab),
            smoothing=Config.CLASS_WEIGHT_SMOOTHING,
        )
        self.class_weights = weights_tensor.to(self.device)

        # 3. Initialize Model
        self.model = MultiGranularityTagger(
            word_vocab_size=len(self.word_vocab),
            char_vocab_size=len(self.char_vocab),
            bpe_vocab_size=Config.BPE_VOCAB_SIZE,
            class_vocab_size=len(self.class_vocab),
            pad_idx=Config.PAD_IDX,
        ).to(self.device)

        # 4. Optimizer & Loss
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, ignore_index=Config.PAD_IDX
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=1
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct_tokens = 0
        total_tokens = 0

        for batch in self.train_loader:
            # Move to device
            word_ids = batch["word_ids"].to(self.device)
            char_ids = batch["char_ids"].to(self.device)
            bpe_ids = batch["bpe_ids"].to(self.device)
            label_ids = batch["label_ids"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward
            logits = self.model(word_ids, char_ids, bpe_ids, mask)

            # Flatten outputs for CrossEntropyLoss
            # logits: (Batch, Seq, Class) -> (Batch*Seq, Class)
            # labels: (Batch, Seq) -> (Batch*Seq)
            loss = self.criterion(
                logits.view(-1, len(self.class_vocab)), label_ids.view(-1)
            )

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Calculate Accuracy
            preds = torch.argmax(logits, dim=-1)
            # Mask out padding from accuracy calculation
            active_mask = label_ids != Config.PAD_IDX
            correct_tokens += (
                (preds == label_ids).masked_select(active_mask).sum().item()
            )
            total_tokens += active_mask.sum().item()

        avg_loss = (
            total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        )
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        return avg_loss, accuracy

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct_tokens = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.val_loader:
                word_ids = batch["word_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                bpe_ids = batch["bpe_ids"].to(self.device)
                label_ids = batch["label_ids"].to(self.device)
                mask = batch["mask"].to(self.device)

                logits = self.model(word_ids, char_ids, bpe_ids, mask)
                loss = self.criterion(
                    logits.view(-1, len(self.class_vocab)), label_ids.view(-1)
                )

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                active_mask = label_ids != Config.PAD_IDX
                correct_tokens += (
                    (preds == label_ids).masked_select(active_mask).sum().item()
                )
                total_tokens += active_mask.sum().item()

        avg_loss = (
            total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        )
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        return avg_loss, accuracy

    def train(self):
        print("\n=== Starting Tagger Training ===")
        best_acc = 0.0
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            end_time = time.time()

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Time: {end_time - start_time:.2f}s"
            )
            print(f"Train Loss: {train_loss} | Train Acc: {train_acc}")
            print(f"Val Loss: {val_loss} | Val Acc: {val_acc}")

            # Scheduler step
            self.scheduler.step(val_acc)

            # Checkpointing & Early Stopping
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.TAGGER_MODEL_PATH)
                print(f"Saved Best Model (Acc: {best_acc})")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(
                        f"Early stopping triggered after {patience_counter} epochs without improvement."
                    )
                    break


class Seq2SeqTrainer:
    """
    Trainer for the Transformer Seq2Seq Fallback Model.
    """

    def __init__(self, debug=Config.DEBUG):
        self.device = get_device()
        self.debug = debug
        set_seed(Config.SEED)

        print(f"Initializing Seq2SeqTrainer (Device: {self.device})...")

        # 1. Load Data
        self.train_loader, self.val_loader, self.char_vocab, self.class_vocab = (
            get_seq2seq_loaders(debug=debug)
        )

        if len(self.train_loader) == 0:
            print("Warning: Seq2Seq Train Loader is empty. Training will be skipped.")
            self.skip_training = True
        else:
            self.skip_training = False

        # 2. Initialize Model
        self.model = TransformerFallback(
            char_vocab_size=len(self.char_vocab),
            class_vocab_size=len(self.class_vocab),
            pad_idx=Config.PAD_IDX,
        ).to(self.device)

        # 3. Optimizer & Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            src_ids = batch["src_ids"].to(self.device)
            tgt_ids = batch["tgt_ids"].to(self.device)
            class_id = batch["class_id"].to(self.device)

            # Teacher Forcing Inputs
            # Input to Decoder: <SOS> ... x_{n-1}
            # Target Labels: x_1 ... <EOS>
            tgt_input = tgt_ids[:, :-1]
            tgt_output = tgt_ids[:, 1:]

            self.optimizer.zero_grad()

            logits = self.model(src_ids, tgt_input, class_id)

            # Loss
            # logits: (Batch, SeqLen, Vocab)
            # targets: (Batch, SeqLen)
            loss = self.criterion(
                logits.reshape(-1, len(self.char_vocab)), tgt_output.reshape(-1)
            )

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                src_ids = batch["src_ids"].to(self.device)
                tgt_ids = batch["tgt_ids"].to(self.device)
                class_id = batch["class_id"].to(self.device)

                tgt_input = tgt_ids[:, :-1]
                tgt_output = tgt_ids[:, 1:]

                logits = self.model(src_ids, tgt_input, class_id)
                loss = self.criterion(
                    logits.reshape(-1, len(self.char_vocab)), tgt_output.reshape(-1)
                )

                total_loss += loss.item()

        return total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0

    def train(self):
        print("\n=== Starting Seq2Seq Training ===")
        if self.skip_training:
            print("Skipping training due to empty dataset.")
            return

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            start_time = time.time()
            train_loss = self.train_epoch()
            val_loss = self.validate()
            end_time = time.time()

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Time: {end_time - start_time:.2f}s"
            )
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")

            self.scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.SEQ2SEQ_MODEL_PATH)
                print(f"Saved Best Model (Loss: {best_loss})")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(
                        f"Early stopping triggered after {patience_counter} epochs without improvement."
                    )
                    break


def train_models(debug=Config.DEBUG):
    """
    Helper function to train both models sequentially.
    """
    # Train Tagger
    tagger_trainer = TaggerTrainer(debug=debug)
    tagger_trainer.train()

    # Train Seq2Seq
    seq2seq_trainer = Seq2SeqTrainer(debug=debug)
    seq2seq_trainer.train()
