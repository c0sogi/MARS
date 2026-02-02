import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.dataset import IcebergDataset, get_data
from library.model import MPDPCNN, train_one_epoch, validate, predict


def run_fold(
    fold_idx,
    total_folds=5,
    epochs=75,
    patience=12,
    batch_size=32,
    lr=1e-3,
    seed=42,
    input_dir="./input",
    metadata_dir="./metadata",
    working_dir="./working/idea_50",
    load_cached_data=True,
):
    """
    Executes the training and validation loop for a single fold of Stratified K-Fold Cross Validation.

    Args:
        fold_idx (int): The index of the current fold (0-based).
        total_folds (int): Total number of folds for CV.
        epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience.
        batch_size (int): Batch size for DataLoaders.
        lr (float): Learning rate.
        seed (int): Random seed for reproducibility.
        input_dir (str): Path to input directory containing JSON files.
        metadata_dir (str): Path to metadata directory containing CSVs.
        working_dir (str): Path to directory for caching data and saving checkpoints.
        load_cached_data (bool): Whether to attempt loading pre-processed data from cache.

    Returns:
        float: The best validation loss achieved for this fold.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure directories exist
    os.makedirs(working_dir, exist_ok=True)
    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- 1. Data Preparation ---
    # We use only the training split for K-Fold CV to preserve the validation set as hold-out
    train_meta_path = os.path.join(metadata_dir, "train.csv")
    raw_json_path = os.path.join(input_dir, "train.json")

    # Determine angle imputation value from the training metadata
    df_train_meta = pd.read_csv(train_meta_path)
    angle_impute_val = df_train_meta["inc_angle"].median()

    # Load original Train split
    X_train_orig, ang_train_orig, y_train_orig, ids_train_orig = get_data(
        "train",
        train_meta_path,
        raw_json_path,
        working_dir,
        load_cached_data,
        angle_impute_val,
    )

    # Use only training data for CV
    X_all = X_train_orig
    ang_all = ang_train_orig
    y_all = y_train_orig
    ids_all = ids_train_orig

    # Generate Stratified K-Fold indices
    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=seed)

    # Retrieve indices for the specific fold
    fold_generator = skf.split(X_all, y_all)
    train_indices, val_indices = next(
        x for i, x in enumerate(fold_generator) if i == fold_idx
    )

    # Subset the data
    X_train, X_val = X_all[train_indices], X_all[val_indices]
    ang_train, ang_val = ang_all[train_indices], ang_all[val_indices]
    y_train, y_val = y_all[train_indices], y_all[val_indices]
    ids_train, ids_val = ids_all[train_indices], ids_all[val_indices]

    # --- 2. Datasets & Loaders ---
    # Define transforms (Random Flip)
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, ids_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, ang_val, y_val, ids_val, transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # --- 3. Model & Optimizer ---
    model = MPDPCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    # Weight decay is handled by AdamW
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # --- 4. Training Loop ---
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting Fold {fold_idx} training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Fold {fold_idx} | Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.10f} - Val Loss: {val_loss:.10f}"
        )

        # Checkpointing & Early Stopping
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_score": best_val_loss,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            checkpoint_dir,
            fold=fold_idx,
        )

        if patience_counter >= patience:
            print(f"Fold {fold_idx} | Early stopping triggered at epoch {epoch+1}")
            break

    return best_val_loss


def generate_submission(
    fold_indices,
    batch_size=32,
    input_dir="./input",
    metadata_dir="./metadata",
    working_dir="./working/idea_50",
    output_dir="./submission",
    load_cached_data=True,
):
    """
    Generates predictions for the test set by ensembling models from multiple folds.

    Args:
        fold_indices (list[int]): List of fold indices to use for ensembling.
        batch_size (int): Batch size for inference.
        input_dir (str): Path to input directory.
        metadata_dir (str): Path to metadata directory.
        working_dir (str): Path to working directory containing checkpoints.
        output_dir (str): Path to save the submission file.
        load_cached_data (bool): Whether to use cached data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load Test Data ---
    test_meta_path = os.path.join(metadata_dir, "test.csv")
    raw_test_json = os.path.join(input_dir, "test.json")

    # We need the angle imputation value from training data to be consistent
    train_meta_path = os.path.join(metadata_dir, "train.csv")
    df_train_meta = pd.read_csv(train_meta_path)
    angle_impute_val = df_train_meta["inc_angle"].median()

    X_test, ang_test, _, ids_test = get_data(
        "test",
        test_meta_path,
        raw_test_json,
        working_dir,
        load_cached_data,
        angle_impute_val,
    )

    test_dataset = IcebergDataset(X_test, ang_test, ids=ids_test, transform=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # --- 2. Inference & Ensembling ---
    # Accumulate probabilities
    avg_probs = np.zeros(len(ids_test))
    valid_folds = 0

    for fold in fold_indices:
        checkpoint_path = os.path.join(
            working_dir, "checkpoints", f"model_best_fold_{fold}.pth"
        )

        if not os.path.exists(checkpoint_path):
            print(
                f"Checkpoint for fold {fold} not found at {checkpoint_path}. Skipping."
            )
            continue

        print(f"Predicting with Fold {fold}...")
        model = MPDPCNN().to(device)

        # Load weights
        load_checkpoint(checkpoint_path, model)

        # Generate predictions
        _, probs = predict(model, test_loader, device)
        avg_probs += np.array(probs)
        valid_folds += 1

    if valid_folds == 0:
        raise RuntimeError("No valid checkpoints found for inference.")

    avg_probs /= valid_folds

    # --- 3. Save Submission ---
    submission_path = os.path.join(output_dir, "submission.csv")
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_probs})
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
