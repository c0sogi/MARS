import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from collections import Counter
from library.config import (
    ProjectConfig,
    TrainingConfig,
    DataConfig,
    ModelConfig,
    set_seed,
)


class Trainer:
    """
    Encapsulates training and evaluation loops for Tagger and Seq2Seq models.
    Handles device management, checkpointing, and metric logging.
    """

    def __init__(self, device=None):
        self.device = device if device else torch.device(TrainingConfig.DEVICE)
        set_seed(TrainingConfig.SEED)

        # Ensure working directory exists for saving models
        os.makedirs(ProjectConfig.BASE_DIR, exist_ok=True)

    def _calculate_class_weights(self, dataset, num_classes):
        """
        Calculates square-root smoothed class weights: weights = sqrt(Total / Count).
        Normalizes weights so the mean is 1.
        """
        # Handle Subset wrapper if present (e.g. from random_split)
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset

        # Get counts
        # TaggerDataset stores classes as a list of strings in self.classes
        if hasattr(dataset, "classes") and dataset.classes is not None:
            counts = Counter(dataset.classes)
        else:
            # Fallback: iterate (slower, generally not needed given TaggerDataset structure)
            counts = Counter()
            # Assuming we can't easily access classes, we return uniform weights
            return torch.ones(num_classes).to(self.device)

        # We need mapping from class name to index
        vocab = dataset.vocab_classes
        total = len(dataset)

        weights = torch.ones(num_classes)

        for cls_name, count in counts.items():
            idx = vocab.stoi.get(cls_name)
            if idx is not None:
                # Square root smoothing
                w = np.sqrt(total / count)
                weights[idx] = w

        # Normalize so mean is 1
        weights = weights / weights.mean()
        return weights.to(self.device)

    def train_tagger(
        self,
        model,
        train_loader,
        val_loader,
        epochs=TrainingConfig.TAGGER_EPOCHS,
        patience=3,
    ):
        """
        Training loop for MultiGranularityTagger.
        """
        print(f"\nStarting Tagger Training for {epochs} epochs...")
        model = model.to(self.device)

        # Configure Loss with Class Weights
        criterion_args = {}
        if TrainingConfig.USE_CLASS_WEIGHTS:
            # Extract dataset from loader
            dataset = train_loader.dataset
            # We assume model.fc.out_features is num_classes based on models.py definition
            num_classes = model.fc.out_features
            try:
                weights = self._calculate_class_weights(dataset, num_classes)
                criterion_args["weight"] = weights
                print("Using Square-Root Smoothed Class Weights.")
            except Exception as e:
                print(
                    f"Warning: Could not calculate class weights ({e}). Using uniform weights."
                )

        criterion = nn.CrossEntropyLoss(**criterion_args)

        optimizer = optim.Adam(
            model.parameters(),
            lr=TrainingConfig.TAGGER_LR,
            weight_decay=TrainingConfig.TAGGER_WEIGHT_DECAY,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=TrainingConfig.SCHEDULER_FACTOR,
            patience=TrainingConfig.SCHEDULER_PATIENCE,
            min_lr=TrainingConfig.SCHEDULER_MIN_LR,
        )

        best_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                word_ids = batch["word_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                targets = batch["class_ids"].to(self.device)

                optimizer.zero_grad()
                logits = model(word_ids, char_ids)
                loss = criterion(logits, targets)

                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), TrainingConfig.TAGGER_GRAD_CLIP
                )
                optimizer.step()

                train_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            avg_train_loss = train_loss / len(train_loader)
            train_acc = correct / total

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    word_ids = batch["word_ids"].to(self.device)
                    char_ids = batch["char_ids"].to(self.device)
                    targets = batch["class_ids"].to(self.device)

                    logits = model(word_ids, char_ids)
                    loss = criterion(logits, targets)

                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total

            epoch_time = time.time() - start_time

            # Print full precision
            print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time}s")
            print(f"Train Loss: {avg_train_loss} | Train Acc: {train_acc}")
            print(f"Val Loss: {avg_val_loss} | Val Acc: {val_acc}")

            scheduler.step(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), ProjectConfig.TAGGER_MODEL_PATH)
                print(f"New best model saved with accuracy: {best_acc}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        return model

    def train_seq2seq(
        self,
        model,
        train_loader,
        val_loader,
        epochs=TrainingConfig.SEQ_EPOCHS,
        patience=3,
    ):
        """
        Training loop for Seq2SeqNormalizer.
        """
        print(f"\nStarting Seq2Seq Training for {epochs} epochs...")
        model = model.to(self.device)

        # Ignore PAD token (index 0)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        optimizer = optim.Adam(
            model.parameters(),
            lr=TrainingConfig.SEQ_LR,
            weight_decay=TrainingConfig.SEQ_WEIGHT_DECAY,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=TrainingConfig.SCHEDULER_FACTOR,
            patience=TrainingConfig.SCHEDULER_PATIENCE,
            min_lr=TrainingConfig.SCHEDULER_MIN_LR,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                src = batch["src_char_ids"].to(self.device)
                tgt = batch["tgt_char_ids"].to(self.device)
                class_ids = batch["class_ids"].to(self.device)

                optimizer.zero_grad()

                # Forward pass with teacher forcing
                outputs = model(
                    src,
                    tgt,
                    class_ids,
                    teacher_forcing_ratio=ModelConfig.TEACHER_FORCING_RATIO,
                )

                # Reshape for loss
                # outputs: (B, MaxLen, Vocab) -> (B*MaxLen, Vocab)
                # tgt: (B, MaxLen) -> (B*MaxLen)
                # The model outputs correspond to steps 1..MaxLen (predictions for next token).
                # The target sequence is [SOS, c1, c2, EOS, PAD].
                # We want to predict [c1, c2, EOS, PAD].
                # So we compare outputs[:, 1:] (predictions) with tgt[:, 1:] (targets).

                output_dim = outputs.shape[-1]
                outputs_sliced = outputs[:, 1:].reshape(-1, output_dim)
                targets_sliced = tgt[:, 1:].reshape(-1)

                loss = criterion(outputs_sliced, targets_sliced)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), TrainingConfig.SEQ_GRAD_CLIP
                )
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    src = batch["src_char_ids"].to(self.device)
                    tgt = batch["tgt_char_ids"].to(self.device)
                    class_ids = batch["class_ids"].to(self.device)

                    # No teacher forcing during validation
                    outputs = model(src, tgt, class_ids, teacher_forcing_ratio=0.0)

                    output_dim = outputs.shape[-1]
                    outputs_sliced = outputs[:, 1:].reshape(-1, output_dim)
                    targets_sliced = tgt[:, 1:].reshape(-1)

                    loss = criterion(outputs_sliced, targets_sliced)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            epoch_time = time.time() - start_time

            # Print full precision
            print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time}s")
            print(f"Train Loss: {avg_train_loss}")
            print(f"Val Loss: {avg_val_loss}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), ProjectConfig.SEQ2SEQ_MODEL_PATH)
                print(f"New best model saved with loss: {best_val_loss}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        return model
