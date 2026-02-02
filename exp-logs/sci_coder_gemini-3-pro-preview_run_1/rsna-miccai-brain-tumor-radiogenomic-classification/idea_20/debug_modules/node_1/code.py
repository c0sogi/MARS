import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

# Import provided library components
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_datasets, WIVSDataset, get_transforms
from library.model import WIVSNet
from library.train import run_fold, predict_test_set


def main():
    # 1. Setup and Configuration
    # ---------------------------------------------------------
    warnings.filterwarnings("ignore")

    # Set a custom working directory for this demo to avoid cache conflicts
    # and ensure we generate data with our specific demo settings (e.g. smaller images)
    Config.WORKING_DIR = "./working/demo_execution_custom"
    Config.setup()  # Re-initialize paths based on new WORKING_DIR

    # Override Config for speed
    Config.IMG_SIZE = 64  # Reduce image size for faster processing
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.NUM_FOLDS = 2  # Run only 2 folds
    Config.BATCH_SIZE = 8  # Small batch size
    Config.DEBUG = True

    print(
        f"Configuration set: Epochs={Config.EPOCHS}, Folds={Config.NUM_FOLDS}, ImgSize={Config.IMG_SIZE}"
    )

    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data Loading & Processing
    # ---------------------------------------------------------
    print("\n[Step 1] Loading and Processing Data...")
    # get_datasets will process DICOMs into 3D volumes (9 channels) and cache them.
    # Since we changed IMG_SIZE and WORKING_DIR, this will run from scratch but be fast due to small size.
    train_ds_full, val_ds_full, test_ds_full = get_datasets(load_cached_data=True)

    # 3. Dataset Truncation (Optimization for Speed)
    # ---------------------------------------------------------
    # To ensure the script runs in < 60 mins (likely < 5 mins), we use a small subset.
    SUBSET_SIZE = 40
    TEST_SUBSET_SIZE = 10

    print(
        f"\n[Step 2] Truncating datasets to {SUBSET_SIZE} samples for demonstration..."
    )

    # Combine train and val to simulate the raw pool before CV split
    all_images = np.concatenate([train_ds_full.images, val_ds_full.images], axis=0)
    all_labels = np.concatenate([train_ds_full.labels, val_ds_full.labels], axis=0)
    all_ids = np.concatenate([train_ds_full.ids, val_ds_full.ids], axis=0)

    # Truncate Train/Val pool
    if len(all_labels) > SUBSET_SIZE:
        all_images = all_images[:SUBSET_SIZE]
        all_labels = all_labels[:SUBSET_SIZE]
        all_ids = all_ids[:SUBSET_SIZE]

    # Truncate Test set
    if len(test_ds_full) > TEST_SUBSET_SIZE:
        test_ds_full.images = test_ds_full.images[:TEST_SUBSET_SIZE]
        test_ds_full.ids = test_ds_full.ids[:TEST_SUBSET_SIZE]
        # test_ds_full.labels is None or dummy, handled internally

    print(
        f"Active Dataset Shapes: Images {all_images.shape}, Labels {all_labels.shape}"
    )

    # 4. Model Verification
    # ---------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")
    # Instantiate model (pretrained=False to avoid download timeouts in restricted envs, though True is default)
    model = WIVSNet(pretrained=False)

    # Assert 1: Check Weight Inflation (First layer should have 9 channels)
    first_conv = model.backbone.conv_stem
    assert (
        first_conv.in_channels == 9
    ), f"Model input channels should be 9, got {first_conv.in_channels}"

    # Assert 2: Check Forward Pass dimensions
    dummy_input = torch.randn(2, 9, Config.IMG_SIZE, Config.IMG_SIZE)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (
        2,
        1,
    ), f"Output shape mismatch. Expected (2, 1), got {output.shape}"

    print("Model verification passed.")

    # 5. Training Loop (Cross-Validation)
    # ---------------------------------------------------------
    print("\n[Step 4] Running Training Loop...")

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels)):
        print(f"\n--- Processing Fold {fold_idx} ---")

        # Split data
        X_train, y_train, ids_train = (
            all_images[train_idx],
            all_labels[train_idx],
            all_ids[train_idx],
        )
        X_val, y_val, ids_val = (
            all_images[val_idx],
            all_labels[val_idx],
            all_ids[val_idx],
        )

        # Create Datasets
        train_fold_ds = WIVSDataset(
            X_train, y_train, ids_train, transform=get_transforms("train")
        )
        val_fold_ds = WIVSDataset(
            X_val, y_val, ids_val, transform=get_transforms("valid")
        )

        # Execute training using the library function
        # This handles model init, training loop, validation, and saving the best model
        best_auc = run_fold(fold_idx, train_fold_ds, val_fold_ds, device)
        fold_scores.append(best_auc)

        # Verify model artifact exists
        model_path = os.path.join(Config.MODEL_DIR, f"wivsnet_fold{fold_idx}.pth")
        assert os.path.exists(model_path), f"Model checkpoint not found at {model_path}"

    print(f"\nCV Scores: {fold_scores}")

    # 6. Inference
    # ---------------------------------------------------------
    print("\n[Step 5] Running Inference on Test Set...")

    # predict_test_set loads the saved models for all folds and averages predictions
    predictions = predict_test_set(test_ds_full, Config.NUM_FOLDS, device)

    # Validation
    assert len(predictions) == len(
        test_ds_full
    ), "Prediction count does not match test set size"
    assert (predictions >= 0).all() and (
        predictions <= 1
    ).all(), "Predictions outside [0, 1] range"

    # 7. Submission
    # ---------------------------------------------------------
    print("\n[Step 6] Creating Submission File...")

    submission_df = pd.DataFrame(
        {"BraTS21ID": test_ds_full.ids, "MGMT_value": predictions}
    )

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Final check
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print(submission_df.head())

    print("\nDemonstration Complete.")


if __name__ == "__main__":
    main()
