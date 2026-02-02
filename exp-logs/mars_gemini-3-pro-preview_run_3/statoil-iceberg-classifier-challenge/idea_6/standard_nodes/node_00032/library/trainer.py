import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.utils import set_seed, get_device
from library.dataset import IcebergDataset, get_dataset, get_transforms
from library.model import SimpleCNN, train_model, predict_with_tta


def run_fold_training(
    fold_idx: int,
    train_dataset: IcebergDataset,
    val_dataset: IcebergDataset,
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 50,
    patience: int = 10,
    save_dir: str = "./working/idea_simple_cnn",
):
    """
    Orchestrates the training process for a single cross-validation fold.

    Args:
        fold_idx (int): The index of the current fold (0-4).
        train_dataset (IcebergDataset): The training dataset for this fold.
        val_dataset (IcebergDataset): The validation dataset for this fold.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        save_dir (str): Directory to save model checkpoints.

    Returns:
        tuple: (trained_model, best_validation_loss)
    """
    # Set seed specific to fold to ensure reproducibility while maintaining diversity across folds
    set_seed(42 + fold_idx)
    device = get_device()

    print(f"--- Starting Training for Fold {fold_idx} ---")

    # Create DataLoaders
    # num_workers=2 is efficient for this dataset size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize Model
    # Cite solution_lesson_node_00019: Lower dropout (0.2) with constant LR
    model = SimpleCNN(dropout_rate=0.2)
    model = model.to(device)

    # Train Model
    # train_model handles the training loop, validation, and early stopping
    model, best_loss = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        patience=patience,
        lr=lr,
        device=device,
    )

    # Save the best model for this fold
    fold_save_dir = os.path.join(save_dir, f"fold_{fold_idx}")
    os.makedirs(fold_save_dir, exist_ok=True)
    save_path = os.path.join(fold_save_dir, "model_best.pth")

    torch.save(model.state_dict(), save_path)
    print(f"Fold {fold_idx} completed. Best Val Loss: {best_loss}")
    print(f"Model saved to: {save_path}")

    return model, best_loss


def train_cross_validation(
    n_folds: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 50,
    patience: int = 10,
    save_dir: str = "./working/idea_simple_cnn",
):
    """
    Runs the full Cross-Validation loop.
    Merges metadata splits and re-splits using StratifiedKFold.
    """
    print(f"Starting {n_folds}-Fold Cross-Validation...")

    # Load existing data splits
    ds_train_part = get_dataset("train", load_cached_data=True)
    ds_val_part = get_dataset("val", load_cached_data=True)

    # Merge data to perform proper K-Fold CV
    X_all = np.concatenate([ds_train_part.X, ds_val_part.X], axis=0)
    angles_all = np.concatenate([ds_train_part.angles, ds_val_part.angles], axis=0)
    y_all = np.concatenate([ds_train_part.labels, ds_val_part.labels], axis=0)

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        # Create datasets for this fold
        # Apply augmentation transforms ONLY to the training set
        train_ds = IcebergDataset(
            X_all[train_idx],
            angles_all[train_idx],
            labels=y_all[train_idx],
            transform=get_transforms("train"),
        )

        # Validation set gets no augmentation
        val_ds = IcebergDataset(
            X_all[val_idx],
            angles_all[val_idx],
            labels=y_all[val_idx],
            transform=get_transforms("val"),
        )

        # Run training for this fold
        _, best_loss = run_fold_training(
            fold_idx,
            train_ds,
            val_ds,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            patience=patience,
            save_dir=save_dir,
        )
        cv_scores.append(best_loss)

    print("-" * 30)
    print(f"CV Complete. Average Log Loss: {np.mean(cv_scores)}")
    print("-" * 30)
    return cv_scores


def generate_submission(
    test_dataset: IcebergDataset = None,
    folds: int = 5,
    batch_size: int = 32,
    model_dir: str = "./working/idea_simple_cnn",
    output_path: str = "./submission/submission.csv",
):
    """
    Generates predictions for the test set using the ensemble of trained models.
    Uses Test-Time Augmentation (TTA).
    """
    set_seed(42)
    device = get_device()

    if test_dataset is None:
        test_dataset = get_dataset("test", load_cached_data=True)

    print(f"Generating submission for {len(test_dataset)} test samples...")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    all_fold_preds = []
    ids = []

    for fold_idx in range(folds):
        model_path = os.path.join(model_dir, f"fold_{fold_idx}", "model_best.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model checkpoint not found at {model_path}. Skipping fold."
            )
            continue

        print(f"Loading model for Fold {fold_idx}...")

        # Initialize and load model
        model = SimpleCNN(dropout_rate=0.2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)

        # Predict with TTA
        fold_ids, fold_preds = predict_with_tta(model, test_loader, device=device)

        # Collect predictions
        all_fold_preds.append(fold_preds)

        # Store IDs from the first successful fold
        if not ids:
            ids = fold_ids

    if not all_fold_preds:
        raise RuntimeError("No valid models found. Cannot generate submission.")

    # Average predictions across all folds (Ensembling)
    avg_preds = np.mean(all_fold_preds, axis=0)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": ids, "is_iceberg": avg_preds})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")
