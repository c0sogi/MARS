import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, verify_initial_loss
from library.dataset import AppleDataset, get_transforms
from library.model import get_model
from library.loss import WeightedSoftTargetCrossEntropy


def get_all_data():
    """
    Loads and merges train and validation metadata to form the full dataset.
    """
    if not os.path.exists(Config.train_metadata_path) or not os.path.exists(
        Config.val_metadata_path
    ):
        raise FileNotFoundError("Metadata files not found.")

    df_train = pd.read_csv(Config.train_metadata_path)
    df_val = pd.read_csv(Config.val_metadata_path)

    # Merge to create full dataset
    df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    return df_full


def get_fixed_split():
    """
    Loads train and validation metadata and combines them with a split indicator.
    Cite solution_lesson_node_00058: Preference for fixed split seed averaging on small datasets.
    """
    if not os.path.exists(Config.train_metadata_path) or not os.path.exists(
        Config.val_metadata_path
    ):
        raise FileNotFoundError("Metadata files not found.")

    df_train = pd.read_csv(Config.train_metadata_path)
    df_val = pd.read_csv(Config.val_metadata_path)

    df_train["split"] = "train"
    df_val["split"] = "valid"

    # Merge to create full dataset
    df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # Ensure stratify_label exists for weight calculation
    if "stratify_label" not in df_full.columns:
        df_full["stratify_label"] = df_full[Config.target_cols].idxmax(axis=1)

    return df_full


def calculate_class_weights(df: pd.DataFrame, device):
    """
    Calculates class weights based on inverse frequency.
    """
    # Count instances per class using the stratify_label
    class_counts = df["stratify_label"].value_counts().sort_index()
    # Ensure order matches Config.target_cols
    counts = np.array([class_counts.get(cls, 0) for cls in Config.target_cols])

    total_samples = len(df)
    n_classes = len(Config.target_cols)

    # Formula: N_total / (N_classes * N_samples_per_class)
    weights = total_samples / (n_classes * counts)

    # Normalize weights so they sum to n_classes (optional, but keeps loss scale similar)
    # or just keep as is. PyTorch CrossEntropy usually takes raw weights.

    print("Class Weights:", dict(zip(Config.target_cols, weights)))

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax for AUC calculation
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate ROC AUC
    # average='macro' calculates metrics for each label, and finds their unweighted mean.
    try:
        val_auc = roc_auc_score(all_targets, all_preds, average="macro")
    except ValueError:
        # Handle edge cases where a class might be missing in validation batch (unlikely with StratifiedKFold)
        val_auc = 0.5

    return val_loss, val_auc


def run_training(run_idx, df, class_weights):
    """
    Runs training for a single seed (Seed Averaging).
    Cite solution_lesson_node_00055: Seed Averaging Ensembles for Robustness.
    """
    current_seed = Config.seeds[run_idx]
    seed_everything(current_seed)
    print(f"\n{'='*20} Running Seed {run_idx} (Seed: {current_seed}) {'='*20}")

    # Split data using fixed split
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "valid"].reset_index(drop=True)

    # Datasets
    train_dataset = AppleDataset(df_train, transform=get_transforms("train"))
    val_dataset = AppleDataset(df_val, transform=get_transforms("valid"))

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Model, Optimizer, Scheduler, Criterion
    model = get_model(Config.model_name, Config.pretrained, Config.num_classes)
    model.to(Config.device)

    optimizer = optim.Adam(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Synced with total epochs
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.epochs, T_mult=Config.T_mult, eta_min=Config.min_lr
    )

    criterion = WeightedSoftTargetCrossEntropy(weight=class_weights)

    # Verify Initial Loss
    print("Verifying initial loss...")
    verify_initial_loss(model, train_loader, criterion, Config.device)

    # Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(
        Config.checkpoint_dir, f"{Config.model_name}_run_{run_idx}.pth"
    )

    for epoch in range(Config.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, Config.device)

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics (Full precision as requested)
        print(f"Epoch {epoch+1}/{Config.epochs} - Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss}")
        print(f"  Val Loss: {val_loss}")
        print(f"  Val AUC: {val_auc}")

        # Checkpointing
        if val_auc > best_auc:
            print(
                f"  [Improvement] AUC increased from {best_auc} to {val_auc}. Saving model..."
            )
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Run {run_idx} finished. Best AUC: {best_auc}")
    return best_auc


def generate_submission(df_test, run_models):
    """
    Generates predictions using an ensemble of seed models.
    """
    print("\nGenerating submission...")

    test_dataset = AppleDataset(
        df_test, transform=get_transforms("test"), test_mode=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Array to store sum of probabilities
    avg_preds = np.zeros((len(df_test), Config.num_classes))

    for run_idx, model_path in enumerate(run_models):
        print(f"Inference with Run {run_idx} model...")

        model = get_model(
            Config.model_name, pretrained=False, num_classes=Config.num_classes
        )
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
        model.to(Config.device)
        model.eval()

        fold_preds = []

        with torch.no_grad():
            for images in test_loader:
                images = images.to(Config.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                fold_preds.append(probs.cpu().numpy())

        fold_preds = np.concatenate(fold_preds, axis=0)
        avg_preds += fold_preds

    # Average predictions
    avg_preds /= len(run_models)

    # Create submission DataFrame
    submission = pd.DataFrame(avg_preds, columns=Config.target_cols)
    submission.insert(0, "image_id", df_test["image_id"])

    # Save
    submission.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")


def train_and_predict():
    """
    Main pipeline function.
    """
    seed_everything(Config.seed)

    # 1. Prepare Data
    df_full = get_fixed_split()

    # 2. Calculate Weights
    class_weights = calculate_class_weights(df_full, Config.device)

    # 3. Train Runs
    run_model_paths = []
    run_scores = []

    for run_idx in range(Config.n_runs):
        score = run_training(run_idx, df_full, class_weights)
        run_scores.append(score)
        run_model_paths.append(
            os.path.join(
                Config.checkpoint_dir, f"{Config.model_name}_run_{run_idx}.pth"
            )
        )

    print("\nRun Scores:", run_scores)
    print("Mean Run AUC:", np.mean(run_scores))

    # 4. Generate Submission
    if os.path.exists(Config.test_metadata_path):
        df_test = pd.read_csv(Config.test_metadata_path)
        generate_submission(df_test, run_model_paths)
    else:
        print("Test metadata not found. Skipping submission.")
