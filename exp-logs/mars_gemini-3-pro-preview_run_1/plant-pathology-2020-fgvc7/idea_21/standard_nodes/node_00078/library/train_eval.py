import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import get_device, seed_everything
from library.dataset import get_data_loaders, get_test_loader
from library.model import AppleResNet34
from library.loss import WeightedSoftCrossEntropy, get_class_weights
from library.sam import SAM


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch using the SAM optimizer.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # SAM requires a closure that calculates the loss
        def closure():
            # Clear gradients (SAM handles the zero_grad logic internally for the steps,
            # but we ensure clean state at the start of the closure execution)
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            return loss

        # Perform optimization step
        # SAM performs two forward-backward passes inside step()
        loss = optimizer.step(closure)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    avg_loss = running_loss / num_samples
    return avg_loss


def validate_one_epoch(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Mean Column-wise ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            # Apply softmax to get probabilities for metric calculation
            probs = torch.softmax(logits, dim=1)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / num_samples

    # Concatenate all batches
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    # Calculate Mean Column-wise ROC AUC
    # We use 'macro' average to treat all classes equally
    try:
        # Check if targets are binary (0/1) or soft.
        # roc_auc_score handles multilabel-indicator format.
        # If targets are soft probabilities, we might need to binarize for strict AUC,
        # but usually passing soft targets works if they represent valid ground truth distributions.
        # Here we assume the metadata provides valid targets.
        auc_score = roc_auc_score(all_targets, all_preds, average="macro")
    except ValueError as e:
        print(
            f"Warning: ROC AUC calculation failed (likely due to missing classes in batch): {e}"
        )
        auc_score = 0.0

    return avg_loss, auc_score


def train_single_fold(split_seed, fold_idx):
    """
    Trains a single model for a specific stratified shuffle-split.

    Args:
        split_seed (int): The seed used for the data split.
        fold_idx (int): Index to identify the model file.
    """
    # 1. Setup
    seed_everything(split_seed)
    device = get_device()
    print(f"\n[Fold {fold_idx} | Seed {split_seed}] Starting training on {device}...")

    # 2. Data Loaders
    # We use the dynamic stratified shuffle-split strategy
    train_loader, val_loader = get_data_loaders(split_seed=split_seed)

    # 3. Model
    model = AppleResNet34(pretrained=Config.PRETRAINED)
    model.to(device)

    # Verify weights
    try:
        model.check_initial_weights()
    except Exception as e:
        print(f"Weight verification failed: {e}")
        return

    # 4. Loss Function
    # Calculate class weights based on the default training metadata
    # (Approximation is sufficient as distribution is stratified)
    df_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    class_weights = get_class_weights(df_meta, Config.CLASS_NAMES, device=device)
    criterion = WeightedSoftCrossEntropy(class_weights=class_weights)

    # 5. Optimizer (SAM)
    # We wrap Adam with SAM
    base_optimizer = torch.optim.Adam
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        rho=Config.SAM_RHO,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 6. Scheduler
    # Cosine Annealing Warm Restarts synchronized to total epochs
    # Note: We must pass the base_optimizer to the scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer.base_optimizer,
        T_0=Config.EPOCHS,  # Single cycle for the whole training duration
        eta_min=Config.MIN_LR,
    )

    # 7. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.MODEL_DIR, f"resnet34_seed_{split_seed}.pth")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )
        val_loss, val_auc = validate_one_epoch(val_loader, model, criterion, device)

        # Update scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val AUC: {val_auc:.16f} - "  # Full precision
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping / Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best AUC! Model saved to {best_model_path}")

    print(f"[Fold {fold_idx}] Finished. Best AUC: {best_auc:.16f}")
    return best_auc


def generate_submission():
    """
    Generates predictions for the test set using the ensemble of trained models.
    Saves the result to submission.csv.
    """
    print("\nStarting Ensemble Inference...")
    device = get_device()
    test_loader = get_test_loader()

    # Initialize array to store sum of probabilities
    # We need to know the number of samples.
    # We'll infer it from the first batch or just collect lists.
    image_ids = []
    ensemble_probs = None

    # Iterate over all seeds/models
    trained_models_count = 0

    for seed in Config.SEEDS:
        model_path = os.path.join(Config.MODEL_DIR, f"resnet34_seed_{seed}.pth")
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        print(f"Loading model from {model_path}...")
        model = AppleResNet34(pretrained=False)  # Weights loaded from state_dict
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        fold_probs = []
        fold_ids = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)

                fold_probs.append(probs.cpu().numpy())
                if trained_models_count == 0:
                    fold_ids.extend(ids)

        fold_probs = np.vstack(fold_probs)

        if ensemble_probs is None:
            ensemble_probs = fold_probs
            image_ids = fold_ids
        else:
            ensemble_probs += fold_probs

        trained_models_count += 1

    if trained_models_count == 0:
        print("Error: No models found for inference.")
        return

    # Average probabilities
    avg_probs = ensemble_probs / trained_models_count

    # Create Submission DataFrame
    df_sub = pd.DataFrame(avg_probs, columns=Config.CLASS_NAMES)
    df_sub.insert(0, "image_id", image_ids)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(df_sub.head())


def run_training_pipeline():
    """
    Orchestrates the full training and submission pipeline.
    """
    # Ensure directories exist
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Train Ensemble
    print(
        f"Starting Stratified Shuffle-Split Ensemble Training ({Config.N_SPLITS} splits)..."
    )

    # In Debug mode, we might reduce splits
    seeds = Config.SEEDS
    if Config.DEBUG:
        seeds = seeds[:2]

    for i, seed in enumerate(seeds):
        train_single_fold(split_seed=seed, fold_idx=i)

    # Generate Submission
    generate_submission()
