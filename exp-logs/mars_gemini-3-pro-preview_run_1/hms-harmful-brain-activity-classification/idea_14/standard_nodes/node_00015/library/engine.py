import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import AverageMeter, kl_divergence, save_checkpoint, seed_everything
from library.models import CyclicFusionNet
from library.dataset import EEGSeizureDataset


def train_one_epoch(loader, model, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch using KL Divergence loss and OneCycleLR scheduler.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (eeg, spec, targets) in enumerate(loader):
        eeg = eeg.to(device, non_blocking=True)
        spec = spec.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        # Modality dropout is applied internally in the model during training
        outputs = model(eeg, spec)

        # Calculate Loss (KL Divergence)
        # Note: library.utils.kl_divergence returns a float (item), so we implement
        # the tensor version here to maintain the gradient graph.
        # F.kl_div expects log-probabilities for input and probabilities for target.
        # We clip predictions to ensure numerical stability.
        epsilon = 1e-15
        outputs_clipped = torch.clamp(outputs, epsilon, 1.0 - epsilon)
        log_preds = torch.log(outputs_clipped)

        loss = F.kl_div(log_preds, targets, reduction="batchmean")

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimization Step
        optimizer.step()

        # Scheduler Step (OneCycleLR steps every batch)
        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), eeg.size(0))

    return losses.avg


def validate(loader, model, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for eeg, spec, targets in loader:
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(eeg, spec)

            # Use the utility function for metric calculation
            loss = kl_divergence(outputs, targets)

            losses.update(loss, eeg.size(0))

    return losses.avg


def inference(loader, model, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for eeg, spec in loader:
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)

            outputs = model(eeg, spec)
            all_preds.append(outputs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def run_training():
    """
    Orchestrates the Cyclic-Subset Training Pipeline.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 1. Data Preparation
    # -------------------------------------------------------------------------
    print("Loading metadata...")
    train_df_full = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    if Config.DEBUG:
        print("DEBUG Mode: Subsampling data...")
        train_df_full = train_df_full.sample(
            n=1000, random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(n=200, random_state=Config.SEED).reset_index(drop=True)

    # Validation Loader (Static)
    val_dataset = EEGSeizureDataset(val_df, mode="val", augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 2. Model Initialization
    # -------------------------------------------------------------------------
    model = CyclicFusionNet()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate steps for OneCycleLR
    # We approximate steps based on average fold size
    avg_fold_size = len(train_df_full) / Config.NUM_FOLDS
    steps_per_epoch = int(np.ceil(avg_fold_size / Config.BATCH_SIZE))

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.TOTAL_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # -------------------------------------------------------------------------
    # 3. Cyclic Training Loop
    # -------------------------------------------------------------------------
    # Create disjoint patient folds
    unique_patients = train_df_full["patient_id"].unique()
    rng = np.random.default_rng(Config.SEED)
    rng.shuffle(unique_patients)
    patient_folds = np.array_split(unique_patients, Config.NUM_FOLDS)

    best_score = float("inf")
    early_stop_counter = 0

    print(
        f"Starting training: {Config.TOTAL_EPOCHS} epochs ({Config.NUM_CYCLES} cycles x {Config.NUM_FOLDS} folds)"
    )

    for epoch in range(Config.TOTAL_EPOCHS):
        # Determine current fold
        fold_idx = epoch % Config.NUM_FOLDS
        cycle_idx = epoch // Config.NUM_FOLDS

        # Create subset for this epoch
        current_patients = patient_folds[fold_idx]
        train_subset = train_df_full[
            train_df_full["patient_id"].isin(current_patients)
        ].copy()

        train_dataset = EEGSeizureDataset(train_subset, mode="train", augment=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        print(
            f"\nEpoch {epoch+1}/{Config.TOTAL_EPOCHS} [Cycle {cycle_idx+1}, Fold {fold_idx+1}] - Samples: {len(train_subset)}"
        )

        # Train
        train_loss = train_one_epoch(
            train_loader, model, optimizer, scheduler, device, epoch
        )

        # Validate
        val_loss = validate(val_loader, model, device)

        print(f"Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}")

        # Save Best Model
        if val_loss < best_score:
            best_score = val_loss
            print(f"New Best Score! Saving model to {Config.MODEL_SAVE_PATH}")
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_score, Config.MODEL_SAVE_PATH
            )
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # -------------------------------------------------------------------------
    # 4. Inference
    # -------------------------------------------------------------------------
    print("\nStarting Inference...")

    # Load Best Model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(
            f"Loaded best model from epoch {checkpoint['epoch']} with score {checkpoint['score']:.6f}"
        )
    else:
        print("Warning: No checkpoint found. Using current model state.")

    test_df = pd.read_csv(Config.TEST_CSV)
    test_dataset = EEGSeizureDataset(test_df, mode="test", augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    preds = inference(test_loader, model, device)

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    sub_df = pd.DataFrame(preds, columns=Config.TARGET_COLS)
    sub_df["eeg_id"] = test_df["eeg_id"]

    # Ensure column order
    cols = ["eeg_id"] + Config.TARGET_COLS
    sub_df = sub_df[cols]

    # Normalize to sum to 1.0 (just in case of float precision issues)
    vote_cols = Config.TARGET_COLS
    sub_df[vote_cols] = sub_df[vote_cols].div(sub_df[vote_cols].sum(axis=1), axis=0)

    sub_df.to_csv(Config.OUTPUT_SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.OUTPUT_SUBMISSION_PATH}")
