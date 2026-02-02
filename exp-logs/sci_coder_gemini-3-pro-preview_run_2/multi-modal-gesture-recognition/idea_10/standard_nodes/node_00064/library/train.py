import os
import time
import torch
import numpy as np
import torch.optim as optim
from library.config import Config
from library.utils import (
    set_seed,
    calculate_levenshtein_accuracy,
    decode_predictions,
    median_filter_prediction,
)
from library.data_loader import get_dataloaders
from library.model import GMD_CRCN
from library.loss import CombinedLoss


class Trainer:
    """
    Trainer class to manage the training and validation lifecycle of the GMD-CRCN model.
    """

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_val_score = float("inf")  # Lower is better for Levenshtein Error Rate

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        running_sub_losses = {}

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            pos = batch["pos"].to(self.device)
            vel = batch["vel"].to(self.device)
            audio = batch["audio"].to(self.device)
            labels = batch["labels"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(pos, vel, audio, lengths)

            # Compute loss
            loss, loss_dict = self.criterion(predictions, labels, lengths)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional but recommended for RNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            # Optimizer step
            self.optimizer.step()

            # Accumulate metrics
            running_loss += loss.item()
            for k, v in loss_dict.items():
                running_sub_losses[k] = running_sub_losses.get(k, 0.0) + v

        end_time = time.time()
        epoch_loss = running_loss / len(self.train_loader)

        # Average sub-losses
        avg_sub_losses = {
            k: v / len(self.train_loader) for k, v in running_sub_losses.items()
        }

        print(
            f"Epoch {epoch} Training Loss: {epoch_loss:.6f} | Time: {end_time - start_time:.2f}s"
        )
        # print(f"    Sub-losses: {avg_sub_losses}")

        return epoch_loss

    def validate(self, epoch):
        """
        Runs validation on the validation set.
        Computes both Loss and Levenshtein Error Rate.
        """
        self.model.eval()
        running_loss = 0.0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                pos = batch["pos"].to(self.device)
                vel = batch["vel"].to(self.device)
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)
                lengths = batch["lengths"].to(self.device)

                # Forward pass
                predictions = self.model(pos, vel, audio, lengths)

                # Compute loss (using Stage 3 output implicitly via CombinedLoss)
                loss, _ = self.criterion(predictions, labels, lengths)
                running_loss += loss.item()

                # Decode predictions for metric calculation
                # Use Stage 3 output: (Batch, Classes, Time)
                logits = predictions["stage3"]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)  # (Batch, Time)

                # Convert to CPU lists
                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                lengths_np = lengths.cpu().numpy()

                for i in range(len(lengths_np)):
                    length = lengths_np[i]
                    # Slice using actual length
                    p_seq = preds_np[i, :length]

                    # Apply Median Filter (Cite solution_lesson_node_00006, solution_lesson_node_00049)
                    p_seq = median_filter_prediction(
                        p_seq, kernel_size=Config.MEDIAN_FILTER_KERNEL
                    )

                    t_seq = labels_np[i, :length]

                    # Decode (collapse repeats, remove background)
                    decoded_p = decode_predictions(
                        p_seq, collapse_repeats=True, remove_background=True
                    )
                    decoded_t = decode_predictions(
                        t_seq, collapse_repeats=True, remove_background=True
                    )

                    all_predictions.append(decoded_p)
                    all_targets.append(decoded_t)

        epoch_loss = running_loss / len(self.val_loader)

        # Compute Levenshtein Error Rate
        error_rate = calculate_levenshtein_accuracy(all_predictions, all_targets)

        print(f"Epoch {epoch} Validation Loss: {epoch_loss:.6f}")
        print(f"Epoch {epoch} Validation Levenshtein Error: {error_rate}")

        return epoch_loss, error_rate

    def fit(self, num_epochs, patience):
        """
        Main training loop with Early Stopping.
        """
        best_epoch = -1
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs with patience {patience}...")

        for epoch in range(1, num_epochs + 1):
            _ = self.train_epoch(epoch)
            val_loss, val_score = self.validate(epoch)

            # Checkpoint logic based on Levenshtein Error Rate
            if val_score < self.best_val_score:
                print(
                    f"New best model found! Score improved from {self.best_val_score} to {val_score}"
                )
                self.best_val_score = val_score
                best_epoch = epoch
                patience_counter = 0

                # Save model
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        print(
            f"Training complete. Best Validation Score: {self.best_val_score} at Epoch {best_epoch}"
        )
        return self.best_val_score


def train_model(
    epochs=Config.NUM_EPOCHS, patience=Config.PATIENCE, batch_size=Config.BATCH_SIZE
):
    """
    Initializes the environment, model, and trainer, then starts training.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loaders
    print("Initializing Data Loaders...")
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # 3. Model
    print("Initializing Model...")
    model = GMD_CRCN().to(device)

    # 4. Loss & Optimizer
    criterion = CombinedLoss(device=device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 5. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    # 6. Run Training
    best_score = trainer.fit(num_epochs=epochs, patience=patience)

    return best_score
