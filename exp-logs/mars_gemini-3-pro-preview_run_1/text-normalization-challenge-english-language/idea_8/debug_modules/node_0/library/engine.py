import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from library.config import Config
from library.utils import save_checkpoint


class Trainer:
    def __init__(self, device=Config.DEVICE):
        self.device = device

    def train_tagger(self, model, train_loader, val_loader, class_weights=None):
        """
        Trains the Bi-LSTM Tagger model.
        """
        model.to(self.device)
        print(f"Starting Tagger training on {self.device}...")

        # Setup Loss with Class Weights
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(
                self.device
            )

        # ignore_index=-100 matches the label padding in TaggerCollator
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-100)

        # Optimizer and Scheduler
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

        for epoch in range(1, Config.TAGGER_EPOCHS + 1):
            start_time = time.time()

            # --- Training Loop ---
            model.train()
            total_loss = 0
            total_correct = 0
            total_tokens = 0

            for batch in train_loader:
                token_ids = batch["token_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                label_ids = batch["label_ids"].to(self.device)
                lengths = batch["lengths"]  # Lengths stay on CPU for packing if needed

                optimizer.zero_grad()

                # Forward pass
                logits = model(token_ids, char_ids, lengths)

                # Permute logits for CrossEntropyLoss: (Batch, Classes, Seq_Len)
                logits = logits.permute(0, 2, 1)

                loss = criterion(logits, label_ids)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Calculate Accuracy
                preds = torch.argmax(logits, dim=1)
                mask = label_ids != -100
                correct = (preds == label_ids) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

            avg_train_loss = total_loss / len(train_loader)
            train_acc = total_correct / total_tokens if total_tokens > 0 else 0.0

            # --- Validation Loop ---
            val_loss, val_acc = self._evaluate_tagger(model, val_loader, criterion)

            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            print(f"Epoch: {epoch:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
            print(
                f"\tTrain Loss: {avg_train_loss:.8f} | Train Acc: {train_acc*100:.6f}%"
            )
            print(f"\t Val. Loss: {val_loss:.8f} |  Val. Acc: {val_acc*100:.6f}%")

            # Scheduler Step (Maximize Accuracy)
            scheduler.step(val_acc)

            # Checkpoint & Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                save_checkpoint(
                    model, optimizer, scheduler, epoch, Config.TAGGER_MODEL_PATH
                )
                print(f"\tNew best model saved!")
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        return best_val_acc

    def _evaluate_tagger(self, model, val_loader, criterion):
        model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                token_ids = batch["token_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                label_ids = batch["label_ids"].to(self.device)
                lengths = batch["lengths"]

                logits = model(token_ids, char_ids, lengths)
                logits = logits.permute(0, 2, 1)

                loss = criterion(logits, label_ids)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                mask = label_ids != -100
                correct = (preds == label_ids) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

        avg_loss = total_loss / len(val_loader)
        acc = total_correct / total_tokens if total_tokens > 0 else 0.0
        return avg_loss, acc

    def train_seq2seq(self, model, train_loader, val_loader):
        """
        Trains the Transformer Seq2Seq Fallback model.
        """
        model.to(self.device)
        print(f"Starting Seq2Seq training on {self.device}...")

        # Loss: Ignore padding
        pad_idx = model.pad_idx
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        # Optimizer and Scheduler
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=True,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, Config.SEQ2SEQ_EPOCHS + 1):
            start_time = time.time()

            # --- Training Loop ---
            model.train()
            total_loss = 0

            for batch in train_loader:
                src_ids = batch["src_ids"].to(self.device)
                tgt_ids = batch["tgt_ids"].to(self.device)
                class_ids = batch["class_ids"].to(self.device)

                # Teacher Forcing
                # Input to decoder: SOS + chars (excluding EOS at end)
                decoder_input = tgt_ids[:, :-1]
                # Target for loss: chars + EOS (excluding SOS at start)
                target = tgt_ids[:, 1:]

                optimizer.zero_grad()

                # Forward pass
                logits = model(src_ids, decoder_input, class_ids)

                # Reshape for loss: (Batch * Seq_Len, Vocab_Size)
                logits = logits.reshape(-1, logits.size(-1))
                target = target.reshape(-1)

                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # --- Validation Loop ---
            val_loss = self._evaluate_seq2seq(model, val_loader, criterion)

            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            print(f"Epoch: {epoch:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
            print(f"\tTrain Loss: {avg_train_loss:.8f}")
            print(f"\t Val. Loss: {val_loss:.8f}")

            # Scheduler Step (Minimize Loss)
            scheduler.step(val_loss)

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    model, optimizer, scheduler, epoch, Config.SEQ2SEQ_MODEL_PATH
                )
                print(f"\tNew best model saved!")
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        return best_val_loss

    def _evaluate_seq2seq(self, model, val_loader, criterion):
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                src_ids = batch["src_ids"].to(self.device)
                tgt_ids = batch["tgt_ids"].to(self.device)
                class_ids = batch["class_ids"].to(self.device)

                decoder_input = tgt_ids[:, :-1]
                target = tgt_ids[:, 1:]

                logits = model(src_ids, decoder_input, class_ids)

                logits = logits.reshape(-1, logits.size(-1))
                target = target.reshape(-1)

                loss = criterion(logits, target)
                total_loss += loss.item()

        return total_loss / len(val_loader)
