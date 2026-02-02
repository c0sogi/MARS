import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter

from library.config import Config
from library.utils import get_device, save_checkpoint
from library.models import PentaHybridTagger, CharLSTMSeq2Seq


class TaggerTrainer:
    """
    Trainer for the PentaHybridTagger (Bi-LSTM).
    """

    def __init__(self, model, train_loader, val_loader, class_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = get_device()
        self.model.to(self.device)

        # Loss Function with Class Weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=True,
        )

    def train(self, num_epochs=Config.NUM_EPOCHS):
        best_acc = 0.0
        patience_counter = 0

        print(f"Starting Tagger training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            start_time = time.time()

            # Training Step
            train_loss, train_acc = self._train_epoch()

            # Validation Step
            val_loss, val_acc = self._validate()

            # Scheduler Step
            self.scheduler.step(val_acc)

            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            print(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
            print(f"\tTrain Loss: {train_loss} | Train Acc: {train_acc}")
            print(f"\t Val. Loss: {val_loss} |  Val. Acc: {val_acc}")

            # Checkpoint & Early Stopping
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_acc": best_acc,
                    },
                    Config.TAGGER_MODEL_PATH,
                )
                print(f"\tNew best model saved!")
            else:
                patience_counter += 1
                print(
                    f"\tEarlyStopping counter: {patience_counter} out of {Config.PATIENCE}"
                )
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch in self.train_loader:
            # Move data to device
            word_ids = batch["word_ids"].to(self.device)
            char_ids = batch["char_ids"].to(self.device)
            bpe_ids = batch["bpe_ids"].to(self.device)
            regex_feats = batch["regex_feats"].to(self.device)
            prior_feats = batch["prior_feats"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Block Feature Dropout (Robustness)
            # Randomly zero out the entire Regex and Prior tensors per sample
            # This forces the model to learn from context (Word/Char/BPE) and not just rely on explicit flags.
            if Config.FEATURE_DROPOUT > 0:
                batch_size = word_ids.size(0)
                # Create mask: (Batch, 1, 1) to broadcast over Sequence and Dim
                # 1.0 means keep, 0.0 means drop
                keep_prob = 1.0 - Config.FEATURE_DROPOUT

                mask_regex = torch.bernoulli(
                    torch.full((batch_size, 1, 1), keep_prob, device=self.device)
                )
                regex_feats = regex_feats * mask_regex

                mask_prior = torch.bernoulli(
                    torch.full((batch_size, 1, 1), keep_prob, device=self.device)
                )
                prior_feats = prior_feats * mask_prior

            self.optimizer.zero_grad()

            # Forward
            logits = self.model(word_ids, char_ids, bpe_ids, regex_feats, prior_feats)

            # Loss Calculation
            # Flatten logits: (Batch * Seq, Num_Classes)
            # Flatten labels: (Batch * Seq)
            loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

            self.optimizer.step()

            epoch_loss += loss.item()

            # Accuracy Calculation
            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100  # Ignore padding/masked tokens
            if mask.sum() > 0:
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()

        avg_loss = epoch_loss / len(self.train_loader)
        avg_acc = correct / total if total > 0 else 0.0
        return avg_loss, avg_acc

    def _validate(self):
        self.model.eval()
        epoch_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                word_ids = batch["word_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                bpe_ids = batch["bpe_ids"].to(self.device)
                regex_feats = batch["regex_feats"].to(self.device)
                prior_feats = batch["prior_feats"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(
                    word_ids, char_ids, bpe_ids, regex_feats, prior_feats
                )

                loss = self.criterion(
                    logits.view(-1, logits.shape[-1]), labels.view(-1)
                )
                epoch_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                mask = labels != -100
                if mask.sum() > 0:
                    correct += (preds[mask] == labels[mask]).sum().item()
                    total += mask.sum().item()

        avg_loss = epoch_loss / len(self.val_loader)
        avg_acc = correct / total if total > 0 else 0.0
        return avg_loss, avg_acc


class FallbackTrainer:
    """
    Trainer for the CharLSTMSeq2Seq Fallback Model.
    """

    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = get_device()
        self.model.to(self.device)

        # 0 is PAD_TOKEN in our Char Vocabulary
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=True,
        )

    def train(self, num_epochs=Config.NUM_EPOCHS):
        best_loss = float("inf")
        patience_counter = 0

        print(f"Starting Fallback Seq2Seq training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss = self._train_epoch()
            val_loss = self._validate()

            self.scheduler.step(val_loss)

            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            print(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
            print(f"\tTrain Loss: {train_loss}")
            print(f"\t Val. Loss: {val_loss}")

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_loss": best_loss,
                    },
                    Config.SEQ2SEQ_MODEL_PATH,
                )
                print(f"\tNew best model saved!")
            else:
                patience_counter += 1
                print(
                    f"\tEarlyStopping counter: {patience_counter} out of {Config.PATIENCE}"
                )
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0

        for batch in self.train_loader:
            src_ids = batch["src_ids"].to(self.device)
            tgt_ids = batch["tgt_ids"].to(self.device)
            class_id = batch["class_id"].to(self.device)

            self.optimizer.zero_grad()

            # Prepare Inputs and Targets
            # tgt_ids: [SOS, c1, c2, ..., EOS, PAD]
            # decoder_input: [SOS, c1, c2, ..., EOS] (remove last PAD if full, or just slice)
            # targets: [c1, c2, ..., EOS, PAD]

            decoder_input = tgt_ids[:, :-1]
            targets = tgt_ids[:, 1:]

            # Forward with Teacher Forcing
            outputs = self.model(
                src_ids, decoder_input, class_id, teacher_forcing_ratio=0.5
            )
            # outputs: (Batch, Seq_Len, Vocab)

            # Loss
            loss = self.criterion(
                outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                src_ids = batch["src_ids"].to(self.device)
                tgt_ids = batch["tgt_ids"].to(self.device)
                class_id = batch["class_id"].to(self.device)

                decoder_input = tgt_ids[:, :-1]
                targets = tgt_ids[:, 1:]

                # Forward without Teacher Forcing (Greedy Decoding for validation)
                # ratio=0.0 means use model predictions
                outputs = self.model(
                    src_ids, decoder_input, class_id, teacher_forcing_ratio=0.0
                )

                loss = self.criterion(
                    outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1)
                )
                epoch_loss += loss.item()

        return epoch_loss / len(self.val_loader)


def compute_class_weights(num_classes):
    """
    Computes square-root smoothed class weights.
    Tries to load from cached labels file first for efficiency.
    """
    print("Computing class weights...")

    # Path where data_loader saves the processed labels
    cache_path = os.path.join(Config.WORKING_DIR, "train_tagger_labels.npy")

    if os.path.exists(cache_path):
        print(f"Loading labels from {cache_path}...")
        try:
            # Load numpy array
            labels = np.load(cache_path)

            # Flatten and remove ignore index (-100) and padding (0 if used as pad)
            # Assuming -100 is the main ignore index used in processing
            valid_mask = labels != -100
            valid_labels = labels[valid_mask]

            # Count
            counts = Counter(valid_labels)

            # Compute Weights: sqrt(Total / Count)
            total_count = len(valid_labels)
            weights = np.ones(num_classes, dtype=np.float32)

            for cls_idx in range(num_classes):
                c = counts.get(cls_idx, 0)
                if c > 0:
                    weights[cls_idx] = np.sqrt(total_count / c)
                else:
                    # If class not present in training batch (unlikely), set weight to 1.0 or high
                    weights[cls_idx] = 1.0

            print("Class weights computed successfully.")
            return torch.tensor(weights)

        except Exception as e:
            print(f"Error computing weights from cache: {e}")
            return None
    else:
        print(f"Cache file {cache_path} not found. Using uniform weights.")
        return None
