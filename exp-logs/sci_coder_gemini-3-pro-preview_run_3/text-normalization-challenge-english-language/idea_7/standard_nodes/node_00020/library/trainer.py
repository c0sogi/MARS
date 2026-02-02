import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.data_utils import get_dataloader, Tokenizer
from library.neural_model import Seq2SeqModel


class Trainer:
    def __init__(self):
        self.set_seed(Config.SEED)
        self.device = Config.DEVICE

        # Initialize Model
        self.model = Seq2SeqModel().to(self.device)

        # Optimizer and Scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1
        )

        # Loss Functions
        # Generation Loss: Ignore PAD_IDX
        self.criterion_gen = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
        # Auxiliary Classification Loss
        self.criterion_aux = nn.CrossEntropyLoss()

        # State for Early Stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def calculate_metrics(self, output_logits, trg_ids):
        """
        Calculates character-level accuracy.
        output_logits: [batch, seq_len, vocab_size]
        trg_ids: [batch, seq_len]
        """
        # Get predictions
        preds = output_logits.argmax(dim=2)  # [batch, seq_len]

        # Create mask for non-pad tokens (we only care about accuracy on real tokens)
        # We also ignore the 0th index (SOS) in the calculation usually,
        # but since we slice both preds and trg starting from 1, it's handled there.
        mask = trg_ids != Config.PAD_IDX

        correct = (preds == trg_ids) & mask

        # Sum correct and total valid tokens
        correct_count = correct.sum().item()
        total_count = mask.sum().item()

        return correct_count, total_count

    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        epoch_loss = 0
        epoch_gen_loss = 0
        epoch_aux_loss = 0

        total_correct = 0
        total_tokens = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            src_char = batch["src_char"].to(self.device)
            src_case = batch["src_case"].to(self.device)
            src_type = batch["src_type"].to(self.device)
            tgt = batch["tgt"].to(self.device)
            class_idx = batch["class_idx"].to(self.device)

            self.optimizer.zero_grad()

            # Forward Pass
            # outputs: [batch, max_len, vocab_size]
            # aux_logits: [batch, num_classes]
            outputs, aux_logits = self.model(
                src_char,
                src_case,
                src_type,
                tgt,
                teacher_forcing_ratio=Config.TEACHER_FORCING_RATIO,
            )

            # Calculate Generation Loss
            # outputs[:, 1:, :] corresponds to predictions for tgt[:, 1:]
            # We discard the 0-th element (SOS)
            output_dim = outputs.shape[-1]

            # Flatten for CrossEntropy
            # outputs: [(batch * (max_len-1)), vocab_size]
            # tgt: [(batch * (max_len-1))]
            outputs_flatten = outputs[:, 1:, :].reshape(-1, output_dim)
            tgt_flatten = tgt[:, 1:].reshape(-1)

            loss_gen = self.criterion_gen(outputs_flatten, tgt_flatten)

            # Calculate Auxiliary Loss
            loss_aux = self.criterion_aux(aux_logits, class_idx)

            # Total Loss
            loss = loss_gen + (Config.LAMBDA_AUX * loss_aux)

            # Backward
            loss.backward()

            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD)

            # Update
            self.optimizer.step()

            # Metrics
            epoch_loss += loss.item()
            epoch_gen_loss += loss_gen.item()
            epoch_aux_loss += loss_aux.item()

            # Accuracy (on the batch)
            c, t = self.calculate_metrics(outputs[:, 1:, :], tgt[:, 1:])
            total_correct += c
            total_tokens += t

        avg_loss = epoch_loss / len(dataloader)
        avg_gen_loss = epoch_gen_loss / len(dataloader)
        avg_aux_loss = epoch_aux_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        print(
            f"Epoch {epoch_idx+1} | Train Loss: {avg_loss:.6f} (Gen: {avg_gen_loss:.6f}, Aux: {avg_aux_loss:.6f}) | Acc: {accuracy:.6f}"
        )
        return avg_loss

    def validate(self, dataloader, epoch_idx):
        self.model.eval()
        epoch_loss = 0
        epoch_gen_loss = 0
        epoch_aux_loss = 0

        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                src_char = batch["src_char"].to(self.device)
                src_case = batch["src_case"].to(self.device)
                src_type = batch["src_type"].to(self.device)
                tgt = batch["tgt"].to(self.device)
                class_idx = batch["class_idx"].to(self.device)

                # Turn off teacher forcing for validation
                outputs, aux_logits = self.model(
                    src_char, src_case, src_type, tgt, teacher_forcing_ratio=0.0
                )

                output_dim = outputs.shape[-1]
                outputs_flatten = outputs[:, 1:, :].reshape(-1, output_dim)
                tgt_flatten = tgt[:, 1:].reshape(-1)

                loss_gen = self.criterion_gen(outputs_flatten, tgt_flatten)
                loss_aux = self.criterion_aux(aux_logits, class_idx)
                loss = loss_gen + (Config.LAMBDA_AUX * loss_aux)

                epoch_loss += loss.item()
                epoch_gen_loss += loss_gen.item()
                epoch_aux_loss += loss_aux.item()

                c, t = self.calculate_metrics(outputs[:, 1:, :], tgt[:, 1:])
                total_correct += c
                total_tokens += t

        avg_loss = epoch_loss / len(dataloader)
        avg_gen_loss = epoch_gen_loss / len(dataloader)
        avg_aux_loss = epoch_aux_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        print(
            f"Epoch {epoch_idx+1} | Val Loss:   {avg_loss:.6f} (Gen: {avg_gen_loss:.6f}, Aux: {avg_aux_loss:.6f}) | Acc: {accuracy:.6f}"
        )
        return avg_loss

    def save_checkpoint(self, path):
        print(f"Saving model checkpoint to {path}")
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        print(f"Loading model checkpoint from {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def fit(self, load_cached_data=True):
        print("Initializing Training Pipeline...")

        # Load Data
        train_loader = get_dataloader(
            "train",
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            load_cached_data=load_cached_data,
        )
        val_loader = get_dataloader(
            "val",
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            load_cached_data=load_cached_data,
        )

        print(
            f"Training on {len(train_loader.dataset)} samples, Validating on {len(val_loader.dataset)} samples."
        )

        for epoch in range(Config.NUM_EPOCHS):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)

            # Scheduler Step
            self.scheduler.step(val_loss)

            # Checkpoint & Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(Config.MODEL_CHECKPOINT_PATH)
            else:
                self.patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {self.patience_counter}/{Config.PATIENCE}"
                )

            end_time = time.time()
            print(f"Epoch Time: {end_time - start_time:.2f}s")

            if self.patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print("Training Complete.")
