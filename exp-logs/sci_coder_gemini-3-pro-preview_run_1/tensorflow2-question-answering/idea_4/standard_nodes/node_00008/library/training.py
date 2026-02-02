import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from library.config import Config, set_seed
from library.utils import (
    setup_logger,
    save_checkpoint,
    compute_classification_metrics,
    compute_span_overlap_f1,
)
from library.data_processing import get_dataloaders
from library.model import DAAN

logger = setup_logger("training")


class MultiTaskLoss(nn.Module):
    """
    Joint loss function for Long Answer (Binary Classification) and
    Short Answer (Span Prediction).
    """

    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        # Long Answer: Binary Cross Entropy
        self.la_loss_fn = nn.BCEWithLogitsLoss()
        # Short Answer: Cross Entropy (Ignore padding/no-answer index -1)
        self.sa_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        la_logits,
        start_logits,
        end_logits,
        la_targets,
        start_targets,
        end_targets,
    ):
        # Long Answer Loss
        loss_la = self.la_loss_fn(la_logits.squeeze(-1), la_targets)

        # Short Answer Loss
        # start_logits: [Batch, MaxC], start_targets: [Batch]
        # Guard against NaN Loss in Sparse Batches (Cite debug_lesson_3)
        valid_start = start_targets != -1
        if valid_start.any():
            loss_start = self.sa_loss_fn(start_logits, start_targets)
        else:
            loss_start = torch.tensor(
                0.0, device=start_logits.device, requires_grad=True
            )

        valid_end = end_targets != -1
        if valid_end.any():
            loss_end = self.sa_loss_fn(end_logits, end_targets)
        else:
            loss_end = torch.tensor(0.0, device=end_logits.device, requires_grad=True)

        # Total Loss
        total_loss = loss_la + loss_start + loss_end
        return total_loss, loss_la, loss_start, loss_end


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0

    all_la_preds = []
    all_la_targets = []

    # For Short Answer metrics (simple approximation for monitoring)
    sa_exact_matches = 0
    sa_total_valid = 0

    with torch.no_grad():
        for batch in dataloader:
            q_input = batch["q_input"].to(device)
            c_input = batch["c_input"].to(device)
            la_targets = batch["label_long"].to(device)
            start_targets = batch["label_short_start"].to(device)
            end_targets = batch["label_short_end"].to(device)

            # Forward pass
            la_logits, start_logits, end_logits = model(q_input, c_input)

            # Compute Loss
            loss, _, _, _ = criterion(
                la_logits,
                start_logits,
                end_logits,
                la_targets,
                start_targets,
                end_targets,
            )
            total_loss += loss.item()

            # Collect Long Answer predictions
            la_probs = torch.sigmoid(la_logits).squeeze(-1).cpu().numpy()
            all_la_preds.append(la_probs)
            all_la_targets.append(la_targets.cpu().numpy())

            # Short Answer Checks (on valid spans only)
            pred_starts = torch.argmax(start_logits, dim=1)
            pred_ends = torch.argmax(end_logits, dim=1)

            # Mask for samples that actually have a short answer
            valid_mask = (start_targets != -1) & (end_targets != -1)
            if valid_mask.sum() > 0:
                matches = (pred_starts[valid_mask] == start_targets[valid_mask]) & (
                    pred_ends[valid_mask] == end_targets[valid_mask]
                )
                sa_exact_matches += matches.sum().item()
                sa_total_valid += valid_mask.sum().item()

    avg_loss = total_loss / len(dataloader)

    # Long Answer Metrics
    all_la_preds = np.concatenate(all_la_preds)
    all_la_targets = np.concatenate(all_la_targets)
    la_metrics = compute_classification_metrics(
        all_la_targets, all_la_preds, threshold=Config.TAU_LONG
    )

    # Short Answer Metrics (Exact Match on valid examples)
    sa_acc = sa_exact_matches / sa_total_valid if sa_total_valid > 0 else 0.0

    return avg_loss, la_metrics, sa_acc


def train_model(load_cached_data=True):
    """
    Main training loop with early stopping.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Data
    logger.info("Loading DataLoaders...")
    train_loader, val_loader, _, embedding_matrix = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    logger.info("Initializing DAAN Model...")
    model = DAAN(embedding_matrix)
    model.to(device)

    # 3. Setup Training Components
    criterion = MultiTaskLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Early Stopping variables
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info("Starting Training...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        # Training Loop
        for batch_idx, batch in enumerate(train_loader):
            q_input = batch["q_input"].to(device)
            c_input = batch["c_input"].to(device)
            la_targets = batch["label_long"].to(device)
            start_targets = batch["label_short_start"].to(device)
            end_targets = batch["label_short_end"].to(device)

            optimizer.zero_grad()

            # Forward
            la_logits, start_logits, end_logits = model(q_input, c_input)

            # Loss
            loss, l_la, l_s, l_e = criterion(
                la_logits,
                start_logits,
                end_logits,
                la_targets,
                start_targets,
                end_targets,
            )

            # Backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 100 == 0 and batch_idx > 0:
                logger.info(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )

        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
        logger.info(f"Validating Epoch {epoch}...")
        val_loss, la_metrics, sa_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        logger.info(f"Epoch {epoch} Summary ({elapsed:.2f}s):")
        logger.info(f"  Train Loss: {avg_train_loss:.6f}")
        logger.info(f"  Val Loss:   {val_loss:.6f}")
        logger.info(
            f"  LA Metrics: Accuracy={la_metrics['accuracy']:.6f}, F1={la_metrics['f1']:.6f}"
        )
        logger.info(f"  SA Metrics: Exact Match (on valid)={sa_acc:.6f}")

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            logger.info(
                f"Validation loss improved from {best_val_loss:.6f} to {val_loss:.6f}. Saving model."
            )
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, Config.MODEL_PATH)
        else:
            patience_counter += 1
            logger.info(
                f"Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

            if patience_counter >= Config.PATIENCE:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete.")
