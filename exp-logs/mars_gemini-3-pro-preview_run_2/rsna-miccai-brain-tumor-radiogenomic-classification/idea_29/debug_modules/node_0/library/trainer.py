import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_CACHE_PATH,
    TRAIN_LABEL_CACHE_PATH,
    VAL_CACHE_PATH,
    VAL_LABEL_CACHE_PATH,
    TEST_CACHE_PATH,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    ROTATION_DEGREES,
    NUM_WORKERS,
    DEVICE,
    SEED,
)
from library.dataset import load_dataset
from library.model import AsymmetricEfficientNet


def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(phase):
    """
    Returns the data transformations for the specified phase.
    Input tensors are (C, H, W).
    """
    if phase == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=ROTATION_DEGREES),
            ]
        )
    else:
        return None


def train_model(debug_max_samples=None):
    """
    Main training loop.
    1. Loads data.
    2. Initializes model, optimizer, loss.
    3. Runs training and validation loops.
    4. Implements early stopping and saves the best model.
    """
    set_seed()
    print(f"Using device: {DEVICE}")

    # --------------------------------------------------------------------------
    # 1. Data Loading
    # --------------------------------------------------------------------------
    print("Initializing Datasets...")

    # Train Dataset
    train_dataset = load_dataset(
        metadata_path=TRAIN_METADATA_PATH,
        cache_path_data=TRAIN_CACHE_PATH,
        cache_path_labels=TRAIN_LABEL_CACHE_PATH,
        load_cached_data=True,
        transform=get_transforms("train"),
        debug_max_samples=debug_max_samples,
    )

    # Validation Dataset
    val_dataset = load_dataset(
        metadata_path=VAL_METADATA_PATH,
        cache_path_data=VAL_CACHE_PATH,
        cache_path_labels=VAL_LABEL_CACHE_PATH,
        load_cached_data=True,
        transform=get_transforms("val"),
        debug_max_samples=debug_max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # 2. Model Setup
    # --------------------------------------------------------------------------
    model = AsymmetricEfficientNet()
    model.to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # --------------------------------------------------------------------------
    # 3. Training Loop
    # --------------------------------------------------------------------------
    best_auc = 0.0
    patience_counter = 0

    print("Starting Training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)  # (B, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = train_loss / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        val_targets = []
        val_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                # labels are already float32 from dataset

                outputs = model(inputs)
                probs = torch.sigmoid(outputs)

                val_preds.extend(probs.cpu().numpy().flatten())
                val_targets.extend(labels.numpy().flatten())

        val_auc = roc_auc_score(val_targets, val_preds)

        print(f"Epoch {epoch}/{NUM_EPOCHS} - Train Loss: {epoch_train_loss:.6f}")
        print(f"Validation AUC: {val_auc}")

        # --- Checkpointing & Early Stopping ---
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")


def predict_and_submit(debug_max_samples=None):
    """
    Inference loop.
    1. Loads test data.
    2. Loads best model.
    3. Predicts using Test-Time Augmentation (TTA).
    4. Generates submission CSV.
    """
    set_seed()
    print("Starting Inference...")

    # 1. Load Test Data
    # Note: Test set has no labels, but dataset returns dummy labels which we ignore
    test_dataset = load_dataset(
        metadata_path=TEST_METADATA_PATH,
        cache_path_data=TEST_CACHE_PATH,
        cache_path_labels=None,  # No labels for test
        load_cached_data=True,
        transform=get_transforms("test"),
        debug_max_samples=debug_max_samples,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Model
    model = AsymmetricEfficientNet()
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_SAVE_PATH}")
    else:
        print(
            f"Warning: Model file not found at {MODEL_SAVE_PATH}. Using random weights."
        )

    model.to(DEVICE)
    model.eval()

    all_preds = []

    # 3. Prediction Loop with TTA
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(DEVICE)

            # TTA 1: Original
            out_orig = model(inputs)
            prob_orig = torch.sigmoid(out_orig)

            # TTA 2: Horizontal Flip (dim 3 is width)
            inputs_h = torch.flip(inputs, dims=[3])
            out_h = model(inputs_h)
            prob_h = torch.sigmoid(out_h)

            # TTA 3: Vertical Flip (dim 2 is height)
            inputs_v = torch.flip(inputs, dims=[2])
            out_v = model(inputs_v)
            prob_v = torch.sigmoid(out_v)

            # Average Probabilities
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0
            all_preds.extend(avg_prob.cpu().numpy().flatten())

    # 4. Generate Submission
    df_test = pd.read_csv(TEST_METADATA_PATH)
    if debug_max_samples is not None:
        df_test = df_test.head(debug_max_samples)

    submission = pd.DataFrame(
        {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": all_preds}
    )

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission.head())
