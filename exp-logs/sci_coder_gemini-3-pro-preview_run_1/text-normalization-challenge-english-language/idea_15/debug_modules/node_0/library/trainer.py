import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from collections import Counter

from library.config import Config
from library.utils import get_logger, calculate_class_weights, weights_to_tensor
from library.models import GatedBiLSTMTagger, Seq2SeqFallback

logger = get_logger("trainer")


class ModelTrainer:
    """
    Handles the training of the Gated Bi-LSTM Tagger and the Seq2Seq Fallback model.
    """

    def __init__(self, device: str = Config.DEVICE):
        self.device = device
        self.checkpoint_dir = Config.CHECKPOINT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def train_tagger(
        self,
        train_dataset,
        val_dataset,
        word_vocab,
        bpe_tokenizer,
        char_vocab,
        class_vocab,
    ):
        """
        Trains the Gated Multi-Granularity Bi-LSTM Tagger.
        """
        logger.info("Setting up Tagger training...")

        # 1. DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 2. Model Initialization
        model = GatedBiLSTMTagger(
            word_vocab_size=len(word_vocab),
            bpe_vocab_size=len(bpe_tokenizer),
            char_vocab_size=len(char_vocab),
            class_vocab_size=len(class_vocab),
            num_regex_feats=Config.NUM_REGEX_FEATURES,
            num_classes=len(class_vocab),
        ).to(self.device)

        # 3. Loss Function with Class Weights
        # Calculate weights based on training data distribution
        logger.info("Calculating class weights...")
        labels = train_dataset.data["labels"].tolist()
        class_counts = Counter([class_vocab.lookup_token(i) for i in labels])
        weights_dict = calculate_class_weights(class_counts)
        class_weights = weights_to_tensor(weights_dict, class_vocab.stoi, self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 4. Optimizer & Scheduler
        optimizer = optim.Adam(model.parameters(), lr=Config.TAGGER_LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=Config.TAGGER_LR_FACTOR,
            patience=Config.TAGGER_LR_PATIENCE,
            verbose=True,
        )

        # 5. Training Loop
        best_val_acc = 0.0
        patience_counter = 0

        logger.info(f"Starting Tagger training for {Config.TAGGER_EPOCHS} epochs...")

        for epoch in range(1, Config.TAGGER_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                # Move inputs to device and unsqueeze to add Seq_Len=1 dimension
                # TaggerDataset returns flat token features, but LSTM expects sequences
                word_ids = batch["word_ids"].to(self.device).unsqueeze(1)
                bpe_ids = batch["bpe_ids"].to(self.device).unsqueeze(1)
                char_ids = batch["char_ids"].to(self.device).unsqueeze(1)
                regex_feats = batch["regex_feats"].to(self.device).unsqueeze(1)
                prior_feats = batch["prior_feats"].to(self.device).unsqueeze(1)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                # Forward pass
                # Output shape: (Batch, Seq_Len=1, Num_Classes)
                logits = model(word_ids, bpe_ids, char_ids, regex_feats, prior_feats)

                # Squeeze to (Batch, Num_Classes) for Loss
                logits = logits.squeeze(1)

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)

                # Accuracy tracking
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            avg_train_loss = total_loss / total
            train_acc = correct / total

            # Validation
            val_acc, val_loss = self._validate_tagger(model, val_loader, criterion)

            # Logging
            logger.info(
                f"Epoch {epoch}/{Config.TAGGER_EPOCHS} | "
                f"Train Loss: {avg_train_loss:.6f} | Train Acc: {train_acc:.6f} | "
                f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
            )

            # Scheduler Step
            scheduler.step(val_acc)

            # Checkpointing & Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save_checkpoint(model, Config.TAGGER_MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= Config.TAGGER_PATIENCE:
                    logger.info("Early stopping triggered for Tagger.")
                    break

        return model

    def _validate_tagger(self, model, loader, criterion):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                word_ids = batch["word_ids"].to(self.device).unsqueeze(1)
                bpe_ids = batch["bpe_ids"].to(self.device).unsqueeze(1)
                char_ids = batch["char_ids"].to(self.device).unsqueeze(1)
                regex_feats = batch["regex_feats"].to(self.device).unsqueeze(1)
                prior_feats = batch["prior_feats"].to(self.device).unsqueeze(1)
                labels = batch["labels"].to(self.device)

                logits = model(word_ids, bpe_ids, char_ids, regex_feats, prior_feats)
                logits = logits.squeeze(1)

                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total, total_loss / total

    def train_seq2seq(
        self,
        train_dataset,
        val_dataset,
        char_vocab,
        class_vocab,
    ):
        """
        Trains the Character-Level LSTM Seq2Seq Fallback Model.
        """
        logger.info("Setting up Seq2Seq training...")

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        model = Seq2SeqFallback(
            char_vocab_size=len(char_vocab),
            class_vocab_size=len(class_vocab),
            sos_idx=char_vocab["<sos>"],
            eos_idx=char_vocab["<eos>"],
            max_seq_len=Config.MAX_SEQ_LEN,
        ).to(self.device)

        # Ignore padding index in loss
        criterion = nn.CrossEntropyLoss(ignore_index=char_vocab["<pad>"])
        optimizer = optim.Adam(model.parameters(), lr=Config.SEQ2SEQ_LR)

        # Scheduler based on Val Loss (min)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1, verbose=True
        )

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Starting Seq2Seq training for {Config.SEQ2SEQ_EPOCHS} epochs...")

        for epoch in range(1, Config.SEQ2SEQ_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            total_tokens = 0

            for src_ids, class_ids, tgt_ids in train_loader:
                src_ids = src_ids.to(self.device)
                class_ids = class_ids.to(self.device)
                tgt_ids = tgt_ids.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                # outputs: (Batch, Seq_Len, Vocab)
                outputs = model(
                    src_ids,
                    class_ids,
                    tgt_ids,
                    teacher_forcing_ratio=Config.SEQ2SEQ_TEACHER_FORCING_RATIO,
                )

                # Reshape for Loss: (Batch, Vocab, Seq_Len) vs (Batch, Seq_Len)
                # We skip the first token of target (<sos>) for loss calculation logic usually,
                # but dataset tgt_ids includes <sos> at start?
                # Dataset: tgt_ids = [<sos>, c1, c2, <eos>, <pad>...]
                # Model output corresponds to prediction for next token.
                # Output at t=0 is prediction for tgt[1].
                # Standard Seq2Seq loss: compare output[t] with tgt[t+1]

                # Model outputs shape is (Batch, Target_Len, Vocab).
                # It generates predictions for positions 0 to Target_Len-1.
                # The prediction at index `t` is predicting `tgt_ids[:, t+1]`.
                # We need to align:
                # Preds: outputs[:, :-1, :] (Drop last prediction which would be for t+1 beyond seq)
                # Targets: tgt_ids[:, 1:] (Drop <sos>)

                output_dim = outputs.shape[-1]
                outputs = outputs[:, :-1, :].reshape(-1, output_dim)
                targets = tgt_ids[:, 1:].reshape(-1)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * targets.size(0)
                total_tokens += targets.size(0)

            avg_train_loss = total_loss / total_tokens

            # Validation
            val_loss = self._validate_seq2seq(model, val_loader, criterion)

            logger.info(
                f"Epoch {epoch}/{Config.SEQ2SEQ_EPOCHS} | "
                f"Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(model, Config.SEQ2SEQ_MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= Config.SEQ2SEQ_PATIENCE:
                    logger.info("Early stopping triggered for Seq2Seq.")
                    break

        return model

    def _validate_seq2seq(self, model, loader, criterion):
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for src_ids, class_ids, tgt_ids in loader:
                src_ids = src_ids.to(self.device)
                class_ids = class_ids.to(self.device)
                tgt_ids = tgt_ids.to(self.device)

                # Teacher forcing 0.0 (or 1.0) doesn't strictly matter for validation loss calculation
                # if we feed targets as input, but usually we validate with what we train (TF)
                # or greedy. For loss calculation, we must feed targets (TF=1.0 implicitly via forward signature).
                # The forward method uses `teacher_forcing_ratio` to decide input.
                # To calculate comparable CrossEntropy, we should use the targets as inputs (TF=1.0)
                # effectively evaluating "Given correct history, what is prob of next char".
                outputs = model(src_ids, class_ids, tgt_ids, teacher_forcing_ratio=1.0)

                output_dim = outputs.shape[-1]
                outputs = outputs[:, :-1, :].reshape(-1, output_dim)
                targets = tgt_ids[:, 1:].reshape(-1)

                loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                total_tokens += targets.size(0)

        return total_loss / total_tokens
