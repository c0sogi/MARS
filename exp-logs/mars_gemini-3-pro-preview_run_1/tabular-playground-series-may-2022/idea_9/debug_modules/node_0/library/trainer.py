import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import time

from library.config import Config
from library.architecture import DiGUT
from library.data_factory import preprocess_data, ManufacturingDataset


def apply_swap_noise(x_num, x_seq, swap_prob):
    """
    Applies swap noise to a batch of data on the GPU.
    Replaces tokens with values from the same feature in a random row within the batch.

    Args:
        x_num: (Batch, Num_Features) FloatTensor
        x_seq: (Batch, Seq_Len) LongTensor
        swap_prob: Probability of swapping a token

    Returns:
        x_num_corr: Corrupted numerical features
        x_seq_corr: Corrupted sequence features
        mask_combined: (Batch, Total_Tokens) Binary mask (1=corrupted, 0=original)
    """
    B = x_num.size(0)
    device = x_num.device

    # Generate random permutation of the batch
    perm = torch.randperm(B, device=device)

    # --- Numerical Corruption ---
    # Create mask: 1 if we should swap, 0 otherwise
    mask_num = (torch.rand(x_num.shape, device=device) < swap_prob).float()
    # Apply swap: mask * permuted + (1-mask) * original
    x_num_corr = mask_num * x_num[perm] + (1 - mask_num) * x_num

    # --- Sequence Corruption ---
    # Create boolean mask
    mask_seq = torch.rand(x_seq.shape, device=device) < swap_prob
    # Apply swap using torch.where for LongTensor
    x_seq_corr = torch.where(mask_seq, x_seq[perm], x_seq)

    # --- Combine Masks ---
    # Concatenate numerical and sequence masks to match discriminator output
    mask_combined = torch.cat([mask_num, mask_seq.float()], dim=1)

    return x_num_corr, x_seq_corr, mask_combined


def train_one_epoch(model, dataloader, optimizer, scheduler, device, config):
    model.train()

    total_loss = 0.0
    total_target_loss = 0.0
    total_disc_loss = 0.0

    # Loss functions
    # Main target loss with label smoothing
    criterion_target = nn.BCEWithLogitsLoss(label_smoothing=config.LABEL_SMOOTHING)
    # Auxiliary discriminator loss
    criterion_disc = nn.BCEWithLogitsLoss()

    for batch_idx, (x_num, x_seq, target, _) in enumerate(dataloader):
        # Move to GPU
        x_num = x_num.to(device)
        x_seq = x_seq.to(device)
        target = target.to(device).unsqueeze(1)  # (B) -> (B, 1)

        # Apply GPU-based Swap Noise
        # Note: We ignore the mask returned by the dataset and generate fresh ones on GPU
        x_num_corr, x_seq_corr, mask_combined = apply_swap_noise(
            x_num, x_seq, config.SWAP_PROB
        )

        optimizer.zero_grad()

        # Forward pass with corrupted data
        target_logits, disc_logits = model(x_num_corr, x_seq_corr)

        # Calculate Losses
        loss_target = criterion_target(target_logits, target)

        # Discriminator loss: Predict mask (1=corrupted)
        # disc_logits: (B, N+S, 1) -> squeeze to (B, N+S)
        loss_disc = criterion_disc(disc_logits.squeeze(-1), mask_combined)

        # Composite Loss
        loss = loss_target + config.AUX_WEIGHT * loss_disc

        # Backward
        loss.backward()

        # Gradient Clipping (optional but good practice for Transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Logging
        total_loss += loss.item()
        total_target_loss += loss_target.item()
        total_disc_loss += loss_disc.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    model.eval()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x_num, x_seq, target, _ in dataloader:
            x_num = x_num.to(device)
            x_seq = x_seq.to(device)

            # No corruption during evaluation
            target_logits, _ = model(x_num, x_seq)

            probs = torch.sigmoid(target_logits).cpu().numpy()
            targets = target.numpy()

            all_preds.append(probs)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5  # Handle edge cases with single class in batch

    return auc, all_preds


def run_training():
    config = Config()
    config.display()

    # Set seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    print(f"Device: {config.DEVICE}")

    # 1. Load and Preprocess Data
    # We use caching to speed up subsequent runs
    (
        X_num_train,
        X_seq_train,
        y_train,
        X_num_val,
        X_seq_val,
        y_val,
        X_num_test,
        X_seq_test,
        ids_test,
        meta,
    ) = preprocess_data(config, load_cached_data=True)

    # 2. Create Datasets and Dataloaders
    # Note: We set is_train=False for train_dataset to bypass CPU-side noise in ManufacturingDataset.
    # We apply noise on GPU in the training loop instead.
    train_dataset = ManufacturingDataset(
        X_num_train, X_seq_train, y_train, is_train=False, config=config
    )
    val_dataset = ManufacturingDataset(
        X_num_val, X_seq_val, y_val, is_train=False, config=config
    )
    test_dataset = ManufacturingDataset(
        X_num_test, X_seq_test, None, is_train=False, config=config
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model
    model = DiGUT(
        num_numerical_features=meta["num_numerical_features"],
        vocab_size=meta["vocab_size"],
        sequence_length=meta["sequence_length"],
        config=config,
    ).to(config.DEVICE)

    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=config.PCT_START,
        anneal_strategy="cos",
    )

    # 5. Training Loop
    best_auc = 0.0
    print("Starting training...")

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, config.DEVICE, config
        )

        val_auc, _ = evaluate(model, val_loader, config.DEVICE)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Loss: {train_loss:.5f} | "
            f"Val AUC: {val_auc:.10f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    print(f"Training complete. Best Val AUC: {best_auc:.10f}")

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    )

    print("Generating predictions on test set...")
    _, test_preds = evaluate(model, test_loader, config.DEVICE)

    # 7. Create Submission
    submission = pd.DataFrame({"id": ids_test, "target": test_preds.flatten()})

    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


if __name__ == "__main__":
    run_training()
