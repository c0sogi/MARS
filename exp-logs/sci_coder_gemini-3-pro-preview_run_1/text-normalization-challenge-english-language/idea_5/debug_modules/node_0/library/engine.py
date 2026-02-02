"""
Implementation of the training and validation engine for the Text Normalization task.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from library.config import Config


class Trainer:
    """
    Handles training and evaluation for Tagger and Seq2Seq models.
    """

    def __init__(self, device=None):
        self.device = device if device else Config.DEVICE

    def get_class_weights(self, df_train, vocab_classes):
        """
        Calculates square-root smoothed class weights based on training data frequency.
        Formula: weight_c = (Total_Tokens / Count_c) ^ alpha
        """
        print("Calculating class weights...")

        # df_train['class'] is a list of lists of class strings in the grouped dataframe
        if "class" not in df_train.columns:
            print("Warning: 'class' column not found in dataframe. Returning None.")
            return None

        # Efficiently count classes
        class_counter = Counter()
        for class_list in df_train["class"]:
            class_counter.update(class_list)

        total_count = sum(class_counter.values())
        num_classes = len(vocab_classes)

        # Initialize weights
        weights = torch.ones(num_classes)

        # Populate weights
        for token, count in class_counter.items():
            idx = vocab_classes.stoi.get(token)
            if idx is not None:
                # Apply smoothing
                weight = (total_count / count) ** Config.CLASS_WEIGHT_SMOOTHING_ALPHA
                weights[idx] = weight

        return weights.to(self.device)

    def train_tagger(self, model, train_loader, val_loader, class_weights=None):
        """
        Trains the Bi-LSTM Tagger model.
        """
        print(f"\nStarting Tagger training on {self.device}...")
        model = model.to(self.device)

        # Loss function with class weights and ignoring padding (index 0)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)

        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=True,
        )

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            total_valid_tokens = 0

            for batch in train_loader:
                word_ids = batch["word_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                class_ids = batch["class_ids"].to(self.device)

                optimizer.zero_grad()

                # Forward pass
                logits = model(word_ids, char_ids)  # (Batch, Seq, Classes)

                # Flatten for loss calculation
                flat_logits = logits.view(-1, logits.shape[-1])
                flat_targets = class_ids.view(-1)

                loss = criterion(flat_logits, flat_targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)
                optimizer.step()

                # Track weighted loss
                num_valid = (flat_targets != 0).sum().item()
                if num_valid > 0:
                    running_loss += loss.item() * num_valid
                    total_valid_tokens += num_valid

            epoch_loss = (
                running_loss / total_valid_tokens if total_valid_tokens > 0 else 0.0
            )

            # Validation
            val_loss, val_acc = self.evaluate_tagger(model, val_loader, criterion)

            print(
                f"Epoch {epoch+1:02d}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {epoch_loss:.8f} | "
                f"Val Loss: {val_loss:.8f} | "
                f"Val Acc: {val_acc:.8f}"
            )

            # Scheduler Step (Monitor Accuracy)
            scheduler.step(val_acc)

            # Checkpointing
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), Config.TAGGER_MODEL_PATH)
                print(f"Saved best Tagger model to {Config.TAGGER_MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(
                        f"Early stopping triggered after {patience_counter} epochs without improvement."
                    )
                    break

        # Restore best model
        if os.path.exists(Config.TAGGER_MODEL_PATH):
            print("Loading best Tagger model...")
            model.load_state_dict(
                torch.load(Config.TAGGER_MODEL_PATH, map_location=self.device)
            )

        return model

    def evaluate_tagger(self, model, val_loader, criterion):
        """
        Evaluates the Tagger model. Returns average loss and accuracy.
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total_valid_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                word_ids = batch["word_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                class_ids = batch["class_ids"].to(self.device)

                logits = model(word_ids, char_ids)

                flat_logits = logits.view(-1, logits.shape[-1])
                flat_targets = class_ids.view(-1)

                loss = criterion(flat_logits, flat_targets)

                # Stats
                num_valid = (flat_targets != 0).sum().item()
                if num_valid > 0:
                    running_loss += loss.item() * num_valid
                    total_valid_tokens += num_valid

                # Accuracy
                preds = torch.argmax(logits, dim=-1)
                mask = class_ids != 0
                correct += (preds[mask] == class_ids[mask]).sum().item()

        avg_loss = running_loss / total_valid_tokens if total_valid_tokens > 0 else 0.0
        accuracy = correct / total_valid_tokens if total_valid_tokens > 0 else 0.0

        return avg_loss, accuracy

    def train_seq2seq(self, model, train_loader, val_loader):
        """
        Trains the Seq2Seq Fallback model.
        """
        print(f"\nStarting Seq2Seq training on {self.device}...")
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.SEQ2SEQ_EPOCHS):
            model.train()
            running_loss = 0.0
            total_tokens = 0  # valid target tokens

            for batch in train_loader:
                src_ids = batch["src_ids"].to(self.device)
                tgt_ids = batch["tgt_ids"].to(self.device)

                optimizer.zero_grad()

                # Forward pass with teacher forcing
                # tgt_ids includes SOS at index 0.
                outputs = model(
                    src_ids,
                    tgt_ids,
                    teacher_forcing_ratio=Config.SEQ2SEQ_TEACHER_FORCING_RATIO,
                )

                # Align outputs and targets
                # outputs[:, t] predicts token at tgt_ids[:, t]
                # Valid predictions are outputs[:, 1:]
                # Targets are tgt_ids[:, 1:]

                output_logits = outputs[:, 1:, :].reshape(-1, outputs.size(-1))
                target_labels = tgt_ids[:, 1:].reshape(-1)

                loss = criterion(output_logits, target_labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)
                optimizer.step()

                # Track loss
                num_valid = (target_labels != 0).sum().item()
                if num_valid > 0:
                    running_loss += loss.item() * num_valid
                    total_tokens += num_valid

            epoch_loss = running_loss / total_tokens if total_tokens > 0 else 0.0

            # Validation
            val_loss = self.evaluate_seq2seq(model, val_loader, criterion)

            print(
                f"Epoch {epoch+1:02d}/{Config.SEQ2SEQ_EPOCHS} | "
                f"Train Loss: {epoch_loss:.8f} | "
                f"Val Loss: {val_loss:.8f}"
            )

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), Config.SEQ2SEQ_MODEL_PATH)
                print(f"Saved best Seq2Seq model to {Config.SEQ2SEQ_MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(
                        f"Early stopping triggered after {patience_counter} epochs without improvement."
                    )
                    break

        # Restore best model
        if os.path.exists(Config.SEQ2SEQ_MODEL_PATH):
            print("Loading best Seq2Seq model...")
            model.load_state_dict(
                torch.load(Config.SEQ2SEQ_MODEL_PATH, map_location=self.device)
            )

        return model

    def evaluate_seq2seq(self, model, val_loader, criterion):
        """
        Evaluates Seq2Seq model using Teacher Forcing (TF=1.0) to calculate Cross Entropy Loss.
        """
        model.eval()
        running_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                src_ids = batch["src_ids"].to(self.device)
                tgt_ids = batch["tgt_ids"].to(self.device)

                # Use TF=1.0 to evaluate likelihood of ground truth
                outputs = model(src_ids, tgt_ids, teacher_forcing_ratio=1.0)

                output_logits = outputs[:, 1:, :].reshape(-1, outputs.size(-1))
                target_labels = tgt_ids[:, 1:].reshape(-1)

                loss = criterion(output_logits, target_labels)

                num_valid = (target_labels != 0).sum().item()
                if num_valid > 0:
                    running_loss += loss.item() * num_valid
                    total_tokens += num_valid

        return running_loss / total_tokens if total_tokens > 0 else 0.0
