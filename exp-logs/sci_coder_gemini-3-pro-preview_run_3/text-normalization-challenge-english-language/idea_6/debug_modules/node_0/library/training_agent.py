import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.data_factory import DataFactory
from library.neural_net import MultiTaskSeq2Seq
from library.utils import set_seed


class Trainer:
    """
    Manages the training lifecycle of the Multi-Task Neuro-Symbolic Cascade model.
    """

    def __init__(self, debug=Config.DEBUG, load_cached_data=True):
        """
        Initializes the Trainer, loads data, and sets up the model.

        Args:
            debug (bool): If True, uses a subset of data for debugging.
            load_cached_data (bool): If True, attempts to load processed data from cache.
        """
        self.debug = debug
        self.device = Config.DEVICE
        set_seed(Config.SEED)

        print(f"Initializing Trainer (Debug={self.debug}, Device={self.device})...")

        # 1. Prepare Data
        self.data_factory = DataFactory()

        # Load Train Loader (fits tokenizer)
        print("Loading Training Data...")
        self.train_loader = self.data_factory.get_train_loader(
            load_cached_data=load_cached_data, debug=debug
        )

        # Load Validation Loader
        print("Loading Validation Data...")
        self.val_loader = self.data_factory.get_val_loader(
            load_cached_data=load_cached_data, debug=debug
        )

        # 2. Initialize Model
        vocab_size = len(self.data_factory.tokenizer)
        print(f"Initializing Model with Vocab Size: {vocab_size}")
        self.model = MultiTaskSeq2Seq(vocab_size).to(self.device)

        # 3. Setup Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Loss Functions
        # Generation: Ignore PAD index
        self.criterion_gen = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
        # Classification: Standard Cross Entropy
        self.criterion_aux = nn.CrossEntropyLoss()

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        total_gen_loss = 0.0
        total_aux_loss = 0.0

        for batch_idx, (src, tgt, cls) in enumerate(self.train_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            cls = cls.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # decoder_outputs: (batch, max_len, vocab)
            # aux_outputs: (batch, num_classes)
            decoder_outputs, aux_outputs = self.model(
                src, tgt, teacher_forcing_ratio=Config.TEACHER_FORCING_RATIO
            )

            # --- Calculate Generation Loss ---
            # We predict the next token.
            # Output at step t corresponds to prediction for target at step t+1.
            # Decoder outputs: [pred_for_t1, pred_for_t2, ..., pred_for_EOS, pred_pad]
            # Targets:         [SOS, t1, t2, ..., EOS]

            # Slice outputs to remove the last step (as we don't have a target for it if length matches)
            # Slice targets to remove the first step (SOS)

            # Note: decoder_outputs length is determined by tgt length in forward pass
            output_preds = decoder_outputs[:, :-1, :].reshape(
                -1, decoder_outputs.shape[-1]
            )
            target_tokens = tgt[:, 1:].reshape(-1)

            loss_gen = self.criterion_gen(output_preds, target_tokens)

            # --- Calculate Auxiliary Loss ---
            loss_aux = self.criterion_aux(aux_outputs, cls)

            # --- Joint Loss ---
            loss = loss_gen + (Config.LAMBDA_AUX * loss_aux)

            # Backward
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD)

            self.optimizer.step()

            # Accumulate
            total_loss += loss.item()
            total_gen_loss += loss_gen.item()
            total_aux_loss += loss_aux.item()

        avg_loss = total_loss / len(self.train_loader)
        avg_gen = total_gen_loss / len(self.train_loader)
        avg_aux = total_aux_loss / len(self.train_loader)

        return avg_loss, avg_gen, avg_aux

    def evaluate(self):
        """
        Evaluates the model on the validation set.
        Calculates Loss and Exact Match Accuracy.
        """
        self.model.eval()
        total_loss = 0.0
        correct_tokens = 0
        total_tokens = 0

        with torch.no_grad():
            for src, tgt, cls in self.val_loader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                cls = cls.to(self.device)

                # Forward pass
                # Use teacher_forcing_ratio=0 for validation to simulate inference
                decoder_outputs, aux_outputs = self.model(
                    src, tgt, teacher_forcing_ratio=0.0
                )

                # --- Calculate Loss (Same logic as train) ---
                output_preds = decoder_outputs[:, :-1, :].reshape(
                    -1, decoder_outputs.shape[-1]
                )
                target_tokens = tgt[:, 1:].reshape(-1)

                loss_gen = self.criterion_gen(output_preds, target_tokens)
                loss_aux = self.criterion_aux(aux_outputs, cls)
                loss = loss_gen + (Config.LAMBDA_AUX * loss_aux)
                total_loss += loss.item()

                # --- Calculate Accuracy ---
                # Decode predictions to strings and compare with target strings
                # decoder_outputs: (batch, seq_len, vocab)
                top1 = decoder_outputs.argmax(2)  # (batch, seq_len)

                # Iterate over batch to decode
                for i in range(src.size(0)):
                    # Decode prediction
                    pred_indices = top1[i]
                    pred_str = self.data_factory.tokenizer.decode(
                        pred_indices, remove_special_tokens=True
                    )

                    # Decode target
                    tgt_indices = tgt[i]
                    tgt_str = self.data_factory.tokenizer.decode(
                        tgt_indices, remove_special_tokens=True
                    )

                    if pred_str == tgt_str:
                        correct_tokens += 1
                    total_tokens += 1

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

        return avg_loss, accuracy

    def fit(self, epochs=Config.NUM_EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs...")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss, train_gen, train_aux = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss} (Gen: {train_gen}, Aux: {train_aux})")

            # Validate
            val_loss, val_acc = self.evaluate()
            print(f"Val Loss: {val_loss}")
            print(f"Val Accuracy: {val_acc}")

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(
                    f"Validation loss improved. Saving model to {Config.MODEL_SAVE_PATH}"
                )
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print("Training complete.")
