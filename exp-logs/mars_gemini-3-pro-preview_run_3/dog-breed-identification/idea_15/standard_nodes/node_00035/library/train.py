import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from library.config import Config
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import DogModel
from library.engine import train_one_epoch, evaluate
from library.utils import seed_everything, average_weights


def train_fold(fold_idx):
    """
    Executes the training pipeline for a single fold, including Head Warmup,
    Fine-Tuning, and Manual Soup generation.

    Args:
        fold_idx (int): The index of the fold to train (0 to N_FOLDS-1).

    Returns:
        str: Path to the saved Soup model for this fold.
    """
    print(f"\n{'='*40}")
    print(f"Starting Fold {fold_idx}")
    print(f"{'='*40}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    # get_dataloaders handles fold splitting and caching internally
    train_loader, val_loader = get_dataloaders(fold_idx)

    # 2. Initialize Model
    # Pretrained=True to load ImageNet-21k weights
    model = DogModel(pretrained=True)
    model.to(device)

    # 3. Phase 1: Head Warmup
    # Freeze backbone to align the random head with pretrained features
    print(f"\n[Fold {fold_idx}] Phase 1: Head Warmup (LR: {Config.LR_WARMUP})")
    model.freeze_backbone()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LR_WARMUP,
        weight_decay=Config.WEIGHT_DECAY,
    )

    for epoch in range(Config.EPOCHS_WARMUP):
        # We pass None for scheduler as it's not used/stepped inside train_one_epoch for warmup
        train_loss = train_one_epoch(
            model, optimizer, None, train_loader, device, epoch + 1
        )
        print(f"Warmup Epoch {epoch + 1} Training Loss: {train_loss}")

    # 4. Phase 2: Fine-Tuning
    # Unfreeze all layers for full network optimization
    print(f"\n[Fold {fold_idx}] Phase 2: Fine-Tuning (LR: {Config.LR_FINE_TUNE})")
    model.unfreeze_all()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR_FINE_TUNE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS_FINE_TUNE, eta_min=Config.MIN_LR
    )

    checkpoint_dir = os.path.join(Config.OUTPUT_DIR, f"fold_{fold_idx}_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    epoch_metrics = []

    for epoch in range(Config.EPOCHS_FINE_TUNE):
        current_epoch = epoch + 1

        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, current_epoch
        )

        # Validate
        val_loss, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {current_epoch} Validation Log Loss: {val_loss}")

        # Step Scheduler
        scheduler.step()

        # Save Checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{current_epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)

        epoch_metrics.append((current_epoch, val_loss, ckpt_path))

    # 5. Create Manual Soup
    print(f"\n[Fold {fold_idx}] Creating Manual Soup...")

    # Sort by validation loss (ascending) to find best epochs
    epoch_metrics.sort(key=lambda x: x[1])

    # Select top K epochs
    top_k_metrics = epoch_metrics[: Config.SOUP_TOP_K]
    print(f"Top {Config.SOUP_TOP_K} Epochs: {[x[0] for x in top_k_metrics]}")
    print(f"Best Val Loss: {top_k_metrics[0][1]}")

    soup_paths = [x[2] for x in top_k_metrics]
    soup_output_path = os.path.join(Config.OUTPUT_DIR, f"best_soup_fold_{fold_idx}.pth")

    # Average weights of the top K models
    average_weights(soup_paths, soup_output_path)

    # Cleanup intermediate checkpoints to conserve storage
    print(f"[Fold {fold_idx}] Cleaning up intermediate checkpoints...")
    for _, _, path in epoch_metrics:
        if os.path.exists(path):
            os.remove(path)

    # Clear memory
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()

    return soup_output_path


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): Test dataloader.
        device (torch.device): Computation device.

    Returns:
        np.ndarray: Averaged probabilities (N, num_classes).
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)

            # 1. Original Image Prediction
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            # 2. Flipped Image Prediction (TTA)
            # Flip width dimension (dim 3 for NCHW)
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped)
            probs_flipped = torch.softmax(logits_flipped, dim=1)

            # 3. Average Probabilities
            avg_probs = (probs + probs_flipped) / 2.0
            all_probs.append(avg_probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def generate_submission():
    """
    Loads all fold soup models, performs TTA inference on the test set,
    ensembles predictions via averaging, and saves the submission file.
    """
    print(f"\n{'='*40}")
    print("Generating Submission")
    print(f"{'='*40}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data
    test_loader, test_df = get_test_dataloader()

    # Load Class Mapping (generated during training/fold preparation)
    classes_path = os.path.join(Config.OUTPUT_DIR, "classes.parquet")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(
            "classes.parquet not found. Run training first to generate metadata."
        )

    classes_df = pd.read_parquet(classes_path)
    # Create mapping: idx -> breed name
    idx_to_class = {row["idx"]: row["breed"] for _, row in classes_df.iterrows()}

    # Initialize Ensemble Predictions Accumulator
    num_test = len(test_df)
    num_classes = Config.NUM_CLASSES
    ensemble_probs = np.zeros((num_test, num_classes))

    valid_folds = 0
    for fold in range(Config.N_FOLDS):
        soup_path = os.path.join(Config.OUTPUT_DIR, f"best_soup_fold_{fold}.pth")

        if not os.path.exists(soup_path):
            print(
                f"Warning: Soup model for fold {fold} not found at {soup_path}. Skipping."
            )
            continue

        print(f"Predicting with Fold {fold} Soup Model...")

        # Load Model
        # We don't need pretrained weights as we are loading a full state dict
        model = DogModel(pretrained=False)
        state_dict = torch.load(soup_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        # Predict with TTA
        probs = predict_with_tta(model, test_loader, device)
        ensemble_probs += probs
        valid_folds += 1

        # Cleanup
        del model, state_dict
        torch.cuda.empty_cache()

    if valid_folds == 0:
        raise RuntimeError("No valid models found for inference.")

    # Average predictions across folds (Bagging)
    ensemble_probs /= valid_folds

    # Create Submission DataFrame
    print("Creating submission file...")
    submission = pd.DataFrame()
    submission["id"] = test_df["id"]

    # Map probabilities to correct breed columns
    for idx in range(num_classes):
        breed_name = idx_to_class[idx]
        submission[breed_name] = ensemble_probs[:, idx]

    # Save to disk
    os.makedirs("submission", exist_ok=True)
    sub_path = os.path.join("submission", "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


def run():
    """
    Main entry point to execute the full Hierarchical Stratified Ensemble pipeline.
    """
    # Setup environment
    Config.setup()

    # Train all folds
    for fold in range(Config.N_FOLDS):
        train_fold(fold)

    # Generate final submission
    generate_submission()
