import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders
from library.model import (
    NarrowResNetMultiScale,
    train_one_epoch,
    validate,
    predict_with_tta,
)


def run_training(
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    n_folds: int = 5,
    patience: int = 5,
    num_workers: int = 2,
    working_dir: str = "./working/idea_16",
    submission_dir: str = "./submission",
    load_cached_data: bool = True,
):
    """
    Orchestrates the training process, evaluation, and submission generation.

    Args:
        epochs (int): Number of training epochs per seed.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        n_folds (int): Number of seeds/folds to train and ensemble.
        patience (int): Early stopping patience epochs.
        num_workers (int): Number of workers for data loaders.
        working_dir (str): Directory to save model checkpoints and cache.
        submission_dir (str): Directory to save the submission file.
        load_cached_data (bool): Whether to attempt loading data from cache.
    """
    # Ensure directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # Load Data
    # get_dataloaders handles the caching logic via load_data internally
    dataloaders = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=load_cached_data,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    test_ids = dataloaders["test_ids"]

    # Store predictions from all seeds
    # Shape: (N_test_samples, N_seeds)
    ensemble_preds = np.zeros((len(test_ids), n_folds))

    for seed in range(n_folds):
        print(f"\n--- Training Seed {seed} ---")
        seed_everything(seed)

        # Initialize Model
        model = NarrowResNetMultiScale().to(device)

        # Loss and Optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Training Loop variables
        best_auc = 0.0
        best_model_path = os.path.join(working_dir, f"model_seed_{seed}.pth")
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validate
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            # Step Scheduler
            scheduler.step()

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss} AUC: {train_auc} | "
                f"Val Loss: {val_loss} AUC: {val_auc}"
            )

            # Early Stopping and Model Checkpointing
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Load best model for this seed for inference
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        else:
            print(
                f"Warning: Best model not found for seed {seed}. Using current weights."
            )

        # Inference with TTA
        print(f"Generating predictions for Seed {seed}...")
        seed_preds = predict_with_tta(model, test_loader, device)
        ensemble_preds[:, seed] = seed_preds

    # Average predictions across seeds (Homogeneous Seed Averaging)
    final_preds = np.mean(ensemble_preds, axis=1)

    # Create submission
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})
    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")
