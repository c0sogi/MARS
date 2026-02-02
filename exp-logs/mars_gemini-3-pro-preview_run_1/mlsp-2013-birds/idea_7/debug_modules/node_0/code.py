import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, save_model, load_model, compute_auc
from library.dataset import BirdDataset, get_dataloaders, load_metadata
from library.model import BirdResNet34
from library.training import Trainer
from library.distillation import generate_ensemble_pseudo_labels

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    print("--- Starting Library Usage Demonstration ---")

    # 1. Configuration Overrides for Demo Speed
    # We modify Config attributes directly to run a fast, lightweight demo.
    print("Configuring for fast demonstration...")
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure reproducibility
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading & Dataset Verification
    print("\n--- Loading Data ---")
    try:
        train_df_full, val_df_full, test_df_full = load_metadata()
    except FileNotFoundError as e:
        print(f"Error loading metadata: {e}")
        return

    # Create small subsets for speed
    subset_size = 16
    train_df = train_df_full.head(subset_size).copy()
    val_df = val_df_full.head(subset_size).copy()
    test_df = test_df_full.head(subset_size).copy()

    print(
        f"Created subsets: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )

    # Instantiate Dataset
    print("Instantiating BirdDataset...")
    train_dataset = BirdDataset(train_df, mode="train")

    # Verify __getitem__
    img, target, rec_id = train_dataset[0]

    # Check shapes
    # Image should be (C, H, W) -> (3, 256, 512)
    assert img.shape == (
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Image shape mismatch. Expected (3, {Config.IMG_HEIGHT}, {Config.IMG_WIDTH}), got {img.shape}"
    # Target should be (NUM_CLASSES,) -> (19,)
    assert target.shape == (
        Config.NUM_CLASSES,
    ), f"Target shape mismatch. Expected ({Config.NUM_CLASSES},), got {target.shape}"

    print("Dataset verification passed: Image and Target shapes are correct.")

    # Create DataLoaders
    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Initialization & Forward Pass
    print("\n--- Initializing Model ---")
    model = BirdResNet34(pretrained=False)  # False to speed up init, logic remains same
    model = model.to(device)

    # Test Forward Pass
    dummy_batch = img.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(dummy_batch)

    assert output.shape == (
        1,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (1, {Config.NUM_CLASSES}), got {output.shape}"
    print("Model forward pass verification passed.")

    # 4. Training Loop Demonstration
    print("\n--- Running Training Loop (1 Epoch) ---")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        criterion=criterion,
        device=device,
    )

    # Fit model
    # We expect this to print log lines (as per Trainer implementation)
    trained_model, history = trainer.fit(num_epochs=Config.EPOCHS)

    # Verify history
    assert (
        "train_loss" in history and len(history["train_loss"]) == 1
    ), "Training history missing or empty."
    assert "val_auc" in history, "Validation AUC missing from history."
    print("Training loop completed successfully.")

    # 5. Model I/O Demonstration
    print("\n--- Testing Model Save/Load ---")
    save_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    save_model(trained_model, save_path)

    assert os.path.exists(save_path), "Model file was not saved."

    # Load model
    new_model = BirdResNet34(pretrained=False).to(device)
    new_model = load_model(new_model, save_path, device=device)
    print("Model saved and loaded successfully.")

    # 6. Distillation / Pseudo-labeling Demonstration
    print("\n--- Testing Distillation (Pseudo-label Generation) ---")
    # We use the trained model as a single 'teacher' for demonstration
    teachers = [trained_model]

    # Force re-generation by ignoring cache if it exists from previous runs
    cache_path = os.path.join(Config.WORKING_DIR, "ensemble_pseudo_labels.parquet")
    if os.path.exists(cache_path):
        os.remove(cache_path)

    pseudo_df = generate_ensemble_pseudo_labels(
        teachers=teachers,
        test_loader=test_loader,
        device=device,
        load_cached_data=False,  # Force generation
    )

    # Verify output
    expected_cols = ["rec_id"] + [f"species_{i}" for i in range(Config.NUM_CLASSES)]
    assert (
        list(pseudo_df.columns) == expected_cols
    ), "Pseudo-label DataFrame columns mismatch."
    assert len(pseudo_df) == len(
        test_df
    ), f"Expected {len(test_df)} predictions, got {len(pseudo_df)}."

    # Check if cache file was created
    assert os.path.exists(cache_path), "Pseudo-label cache file was not created."
    print("Pseudo-label generation verified.")

    # 7. Metric Utility Verification
    print("\n--- Verifying Metric Calculation (AUC) ---")
    # Case 1: Perfect prediction
    y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    y_pred_perfect = np.array([[0.1, 0.9, 0.1], [0.9, 0.1, 0.1], [0.1, 0.1, 0.9]])

    # Note: compute_auc expects N classes. Our config has 19.
    # The function iterates up to y_true.shape[1]. So we can pass 3 classes for testing logic.
    score_perfect = compute_auc(y_true, y_pred_perfect)
    assert (
        score_perfect == 1.0
    ), f"Expected AUC 1.0 for perfect predictions, got {score_perfect}"

    # Case 2: Random/Bad prediction
    y_pred_bad = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9], [0.9, 0.9, 0.1]])
    score_bad = compute_auc(y_true, y_pred_bad)
    assert score_bad < 1.0, "Expected AUC < 1.0 for bad predictions."

    print(
        f"Metric verification passed. Perfect Score: {score_perfect}, Bad Score: {score_bad}"
    )

    print("\n--- Demonstration Complete ---")


if __name__ == "__main__":
    main()
