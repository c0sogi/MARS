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
from library.architecture import DeGUT
from library.data_factory import preprocess_data, ManufacturingDataset


def train_one_epoch(model, dataloader, optimizer, scheduler, device, config):
    model.train()

    total_loss = 0.0
    total_target_loss = 0.0
    total_aux_loss = 0.0

    # Loss functions
    # Main target loss with label smoothing
    criterion_target = nn.BCEWithLogitsLoss(label_smoothing=config.LABEL_SMOOTHING)
    # Reconstruction losses
    criterion_num = nn.MSELoss()
    criterion_seq = nn.CrossEntropyLoss()

    for batch_idx, (x_num, x_seq, target, _) in enumerate(dataloader):
        # Move to GPU
        x_num = x_num.to(device)
        x_seq = x_seq.to(device)
        target = target.to(device).unsqueeze(1)

        # Generate Masks on GPU
        mask_num = torch.rand(x_num.shape, device=device) < config.MASK_PROB
        mask_seq = torch.rand(x_seq.shape, device=device) < config.MASK_PROB

        optimizer.zero_grad()

        # Forward pass with masks
        target_logits, num_preds, seq_preds = model(x_num, x_seq, mask_num, mask_seq)

        # Calculate Losses
        loss_target = criterion_target(target_logits, target)

        # Auxiliary Reconstruction Loss (only on masked tokens)
        loss_num = 0.0
        if mask_num.any():
            loss_num = criterion_num(num_preds[mask_num], x_num[mask_num])

        loss_seq = 0.0
        if mask_seq.any():
            loss_seq = criterion_seq(seq_preds[mask_seq], x_seq[mask_seq])

        loss_aux = loss_num + loss_seq
        loss = loss_target + config.AUX_WEIGHT * loss_aux

        # Backward
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Logging
        total_loss += loss.item()
        total_target_loss += loss_target.item()
        total_aux_loss += (
            loss_aux.item() if isinstance(loss_aux, torch.Tensor) else loss_aux
        )

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

            # No masking during evaluation
            target_logits, _, _ = model(x_num, x_seq)

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
    model = DeGUT(
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
