import os
import time
import torch
import torch.optim as optim
import numpy as np
import scipy.ndimage
from library.config import Config
from library.utils import (
    set_seed,
    AverageMeter,
    compute_levenshtein,
    save_checkpoint,
    load_checkpoint,
)
from library.data_loader import get_dataloaders
from library.model import DSG_CRCN
from library.loss import DeepSupervisionLoss


def decode_sequence(frame_indices, kernel_size=7):
    """
    Decodes frame-wise indices into a sequence of gesture labels.
    1. Applies Median Filter to smooth noise.
    2. Collapses consecutive duplicates.
    3. Removes background class (0).
    """
    # 1. Median Filter
    if kernel_size > 1 and len(frame_indices) >= kernel_size:
        smoothed = scipy.ndimage.median_filter(frame_indices, size=kernel_size)
    else:
        smoothed = frame_indices

    # 2. Collapse Repeats & 3. Remove Background
    sequence = []
    last_val = -1

    for val in smoothed:
        if val != last_val:
            if val != 0:  # 0 is background
                sequence.append(int(val))
            last_val = val

    return sequence


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = config.DEVICE

    def train_one_epoch(self, epoch):
        self.model.train()

        losses = AverageMeter()
        batch_time = AverageMeter()

        start_time = time.time()

        for i, batch in enumerate(self.train_loader):
            # Unpack batch
            features = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            boundaries = batch[2].to(self.device)
            mask = batch[3].to(self.device)
            # ids = batch[4] # Not needed for training

            # Forward pass
            outputs = self.model(features, mask)

            # Compute Loss
            loss, loss_metrics = self.criterion(outputs, targets, boundaries, mask)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping (Optional but recommended for RNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            # Update metrics
            losses.update(loss.item(), features.size(0))

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            # Log occasionally
            if i % 10 == 0:
                # Construct metric string
                metric_str = " ".join([f"{k}:{v:.4f}" for k, v in loss_metrics.items()])
                # print(f"Epoch [{epoch}][{i}/{len(self.train_loader)}] "
                #       f"Loss {losses.val:.4f} ({losses.avg:.4f}) {metric_str}")

        return losses.avg

    def validate(self):
        self.model.eval()

        losses = AverageMeter()
        levenshtein_score = AverageMeter()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                boundaries = batch[2].to(self.device)
                mask = batch[3].to(self.device)

                # Forward pass
                outputs = self.model(features, mask)

                # Compute Loss
                loss, _ = self.criterion(outputs, targets, boundaries, mask)
                losses.update(loss.item(), features.size(0))

                # Inference / Decoding
                # Use Stage 3 Class Probabilities
                final_cls = outputs["final_cls"]  # (B, T, C)

                # Get hard predictions
                _, pred_indices = torch.max(final_cls, dim=2)  # (B, T)

                # Move to CPU
                pred_indices = pred_indices.cpu().numpy()
                targets_np = targets.cpu().numpy()
                mask_np = mask.cpu().numpy()

                # Decode each sequence in batch
                for b in range(features.size(0)):
                    # Get valid length
                    valid_len = np.sum(mask_np[b])

                    # Slice valid frames
                    p_seq_raw = pred_indices[b, :valid_len]
                    t_seq_raw = targets_np[b, :valid_len]

                    # Decode
                    p_seq_decoded = decode_sequence(p_seq_raw, kernel_size=7)
                    t_seq_decoded = decode_sequence(
                        t_seq_raw, kernel_size=1
                    )  # No filtering needed for GT

                    all_preds.append(p_seq_decoded)
                    all_targets.append(t_seq_decoded)

        # Compute Metric
        score = compute_levenshtein(all_preds, all_targets)

        return losses.avg, score

    def fit(self, epochs, patience):
        best_score = float("inf")
        patience_counter = 0

        print(f"[+] Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_score = self.validate()

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Levenshtein: {val_score:.6f}"
            )

            # Checkpointing
            is_best = val_score < best_score
            if is_best:
                best_score = val_score
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_score": best_score,
                    },
                    is_best=True,
                )
                print(f"    -> New best model saved! Score: {best_score:.6f}")
            else:
                patience_counter += 1
                print(f"    -> Patience: {patience_counter}/{patience}")

            # Early Stopping
            if patience_counter >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch}")
                break

        return best_score


def train_model(load_cached_data=True, epochs=Config.EPOCHS, patience=Config.PATIENCE):
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)

    # 2. Data
    print("[*] Loading data...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Model
    print("[*] Initializing DSG-CRCN model...")
    model = DSG_CRCN().to(Config.DEVICE)

    # 4. Loss & Optimizer
    criterion = DeepSupervisionLoss().to(Config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=Config,
    )

    # 6. Run
    best_score = trainer.fit(epochs=epochs, patience=patience)
    print(f"[*] Training finished. Best Validation Levenshtein Score: {best_score:.6f}")

    return trainer
