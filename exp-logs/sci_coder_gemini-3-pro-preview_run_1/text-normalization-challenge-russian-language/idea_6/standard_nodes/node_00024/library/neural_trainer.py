import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Tuple
import time
import math

from library.config import Config, DEVICE, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from library.neural_arch import DualGranularityTransformer
from library.tokenizers import HybridTokenizer
from library.utils import safe_save_model, print_metrics, ensure_dir


class ModelTrainer:
    """
    Manages the training lifecycle of the DualGranularityTransformer.
    """

    def __init__(
        self,
        config: Config,
        model: DualGranularityTransformer,
        tokenizer: HybridTokenizer,
    ):
        self.config = config
        self.model = model.to(DEVICE)
        self.tokenizer = tokenizer

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=1e-6
        )

        # Loss Function
        # We ignore the padding token in the loss calculation
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=config.label_smoothing,
        )

        self.best_val_accuracy = -1.0
        self.best_val_loss = float("inf")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {DEVICE}")
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()

            # --- Training Step ---
            train_loss = self._train_epoch(train_loader)

            # --- Validation Step ---
            val_loss, val_accuracy = self._validate(val_loader)

            # --- Scheduler Step ---
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # --- Logging ---
            elapsed = time.time() - start_time
            metrics = {
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Val Accuracy": val_accuracy,
                "LR": current_lr,
                "Time": f"{elapsed:.2f}s",
            }
            print_metrics(metrics)

            # --- Checkpointing & Early Stopping ---
            # We prioritize Accuracy as it is the competition metric
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_val_loss = val_loss
                patience_counter = 0
                print(
                    f"New best accuracy! Saving model to {self.config.model_checkpoint_path}"
                )
                safe_save_model(
                    self.model.state_dict(), self.config.model_checkpoint_path
                )
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

    def _train_epoch(self, loader: DataLoader) -> float:
        """
        Runs one epoch of training with Gradient Accumulation.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            # Move batch to device
            src_left = batch["src_left"].to(DEVICE)
            src_target = batch["src_target"].to(DEVICE)
            src_right = batch["src_right"].to(DEVICE)
            tgt = batch["tgt"].to(DEVICE)  # Contains [SOS, ..., EOS]

            # Prepare Inputs and Targets for Teacher Forcing
            # Input to decoder: [SOS, t1, t2, ...] (exclude last EOS)
            tgt_input = tgt[:, :-1]
            # Target for loss: [t1, t2, ..., EOS] (exclude first SOS)
            tgt_output = tgt[:, 1:]

            # Forward Pass
            logits = self.model(src_left, src_target, src_right, tgt_input)

            # Reshape for Loss
            # logits: [Batch, Seq_Len, Vocab] -> [Batch*Seq_Len, Vocab]
            # tgt_output: [Batch, Seq_Len] -> [Batch*Seq_Len]
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
            )

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

            # Backward Pass
            loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0 or (
                batch_idx + 1
            ) == len(loader):
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

                self.optimizer.step()
                self.optimizer.zero_grad()

            # Track total loss (rescale back to original magnitude for logging)
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Runs validation. Computes Loss (Teacher Forcing) and Accuracy (Greedy Decoding).
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        # We use a subset for accuracy if validation set is huge to save time,
        # but here we'll try to run full validation as accuracy is critical.
        with torch.no_grad():
            for batch in loader:
                src_left = batch["src_left"].to(DEVICE)
                src_target = batch["src_target"].to(DEVICE)
                src_right = batch["src_right"].to(DEVICE)
                tgt = batch["tgt"].to(DEVICE)
                original_after = batch["original_after"]

                # --- 1. Calculate Loss ---
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                logits = self.model(src_left, src_target, src_right, tgt_input)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
                )
                total_loss += loss.item()
                num_batches += 1

                # --- 2. Calculate Accuracy (Greedy Decoding) ---
                # This simulates inference
                predictions = self._greedy_decode_batch(src_left, src_target, src_right)

                # Compare strings
                for pred, truth in zip(predictions, original_after):
                    if pred == truth:
                        total_correct += 1
                total_samples += len(original_after)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy

    def _greedy_decode_batch(
        self, src_left: torch.Tensor, src_target: torch.Tensor, src_right: torch.Tensor
    ) -> List[str]:
        """
        Performs greedy decoding for a batch of inputs.
        """
        batch_size = src_left.size(0)
        max_len = self.config.max_seq_len

        # 1. Encode
        memory, memory_mask = self.model.encode(src_left, src_target, src_right)

        # 2. Initialize Decoder Input with SOS
        sos_id = self.tokenizer.bpe_tokenizer.token_to_id(SOS_TOKEN)
        eos_id = self.tokenizer.bpe_tokenizer.token_to_id(EOS_TOKEN)

        # [Batch, 1]
        ys = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=DEVICE)

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

        for _ in range(max_len):
            # Decode
            # Note: memory_mask is the src_key_padding_mask
            out = self.model.decode(ys, memory, memory_key_padding_mask=memory_mask)

            # Get last token logits: [Batch, 1, Vocab]
            prob = out[:, -1, :]

            # Greedy choice
            _, next_word = torch.max(prob, dim=1)

            # Append
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # Update finished status
            finished |= next_word == eos_id

            if finished.all():
                break

        # 3. Convert IDs to Strings
        # Remove SOS (first token) and stop at EOS
        predictions = []
        ys_list = ys.tolist()

        for row in ys_list:
            # Skip SOS at index 0
            tokens = []
            for token_id in row[1:]:
                if token_id == eos_id:
                    break
                tokens.append(token_id)

            decoded_str = self.tokenizer.decode(tokens, skip_special_tokens=True)
            predictions.append(decoded_str)

        return predictions
