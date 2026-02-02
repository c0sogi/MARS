import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.dataset import load_data, RetinopathyDataset, get_transforms
from library.model import RetinopathyRegressor
from library.engine import run_training
from torch.utils.data import DataLoader


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    print("Setting up configuration...")

    # Override Config for a fast demonstration run
    # We use a separate working directory to ensure clean cache generation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples for speed
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size for debug
    Config.WORKING_DIR = "./working/demo_execution"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model_best.pth")
    Config.SUBMISSION_SAVE_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Initialize environment (creates directories)
    Config.setup()
    seed_everything(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\nLoading data...")
    # load_data handles caching and splitting based on Config
    train_df, val_df, test_df = load_data(load_cached_data=False)

    # Verify data loading
    assert len(train_df) > 0, "Train DataFrame is empty"
    assert len(val_df) > 0, "Validation DataFrame is empty"
    assert len(test_df) > 0, "Test DataFrame is empty"
    print(
        f"Data Loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # ==========================================
    # 3. Dataset and Dataloader Creation
    # ==========================================
    print("\nCreating datasets and dataloaders...")

    train_dataset = RetinopathyDataset(
        df=train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = RetinopathyDataset(
        df=val_df, transforms=get_transforms("valid"), mode="valid"
    )
    test_dataset = RetinopathyDataset(
        df=test_df, transforms=get_transforms("test"), mode="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Verify Batch Shape
    dummy_images, dummy_targets = next(iter(train_loader))
    assert dummy_images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {dummy_images.shape}"
    assert dummy_targets.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect target shape: {dummy_targets.shape}"
    print("DataLoader shapes verified.")

    # ==========================================
    # 4. Model Initialization
    # ==========================================
    print("\nInitializing model...")
    model = RetinopathyRegressor(pretrained=True)
    model.to(Config.DEVICE)

    # Verify Model Output logic
    with torch.no_grad():
        dummy_input = dummy_images.to(Config.DEVICE)
        dummy_output = model(dummy_input)
        # Expected shape is (Batch_Size,) for regression
        assert dummy_output.shape == (
            Config.BATCH_SIZE,
        ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {dummy_output.shape}"
    print("Model output shape verified.")

    # ==========================================
    # 5. Training
    # ==========================================
    print("\nStarting training loop...")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run the training engine
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=Config.DEVICE,
        num_epochs=Config.NUM_EPOCHS,
        patience=2,
    )

    # ==========================================
    # 6. Inference and Submission
    # ==========================================
    print("\nRunning inference on test set...")

    # Load best model if saved
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )
    else:
        print(
            "No checkpoint saved (likely no improvement). Using current model weights."
        )

    model.eval()
    test_preds = []
    test_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(Config.DEVICE, dtype=torch.float)
            outputs = model(images)

            # Move to CPU
            outputs = outputs.cpu().numpy()

            # Post-processing for Regression -> Classification
            # 1. Clip to valid range [0, 4]
            outputs = np.clip(outputs, 0, 4)
            # 2. Round to nearest integer
            outputs = np.round(outputs).astype(int)

            test_preds.extend(outputs)
            test_ids.extend(ids)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": test_preds})

    # Verify submission content
    assert len(submission_df) == len(test_df), "Submission row count mismatch"
    # Check that all predictions are valid integers 0-4
    unique_preds = set(submission_df["diagnosis"].unique())
    assert unique_preds.issubset(
        {0, 1, 2, 3, 4}
    ), f"Invalid diagnosis values in submission: {unique_preds}"

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_SAVE_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_SAVE_PATH}")
    print("First 5 rows of submission:")
    print(submission_df.head())

    # ==========================================
    # 7. Metric Logic Verification (Self-Check)
    # ==========================================
    print("\nVerifying metric logic...")
    # Create dummy ground truth and predictions
    y_true_dummy = [0, 1, 2, 3, 4]
    y_pred_perfect = [0, 1, 2, 3, 4]
    y_pred_bad = [4, 3, 2, 1, 0]

    score_perfect = quadratic_weighted_kappa(y_true_dummy, y_pred_perfect)
    score_bad = quadratic_weighted_kappa(y_true_dummy, y_pred_bad)

    assert np.isclose(
        score_perfect, 1.0
    ), "Metric check failed: Perfect score should be 1.0"
    assert score_bad < 0.5, "Metric check failed: Bad score should be low"
    print(
        f"Metric verification passed. Perfect Score: {score_perfect}, Bad Score: {score_bad:.4f}"
    )

    print("\nJob complete.")


if __name__ == "__main__":
    main()
