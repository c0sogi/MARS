import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import AppleLeafDataset, get_transforms
from library.models import get_model
from library.train import run_training_fold
from library.inference import generate_submission


def main():
    print("==== Apple Disease Detection Pipeline Demonstration ====\n")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("[1] Setting up configuration for demo run...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.N_FOLDS = 2  # Run only 2 folds
    Config.BATCH_SIZE = 8  # Small batch size
    Config.IMG_SIZE = 128  # Smaller image size for faster processing
    Config.MODEL_ARCHS = ["resnet34"]  # Use the lighter model only

    # Define working directories for this demo
    DEMO_DIR = "./working/demo_execution"
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    Config.create_dirs()

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # ==========================================
    # 2. Data Preparation (Subsets)
    # ==========================================
    print("\n[2] Preparing data subsets...")

    # Load original metadata
    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    full_val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create small subsets (20 samples each) to simulate a quick training run
    train_subset = full_train_df.sample(n=20, random_state=Config.SEED).reset_index(
        drop=True
    )
    val_subset = full_val_df.sample(n=20, random_state=Config.SEED).reset_index(
        drop=True
    )

    print(f"    Train subset shape: {train_subset.shape}")
    print(f"    Valid subset shape: {val_subset.shape}")

    # ==========================================
    # 3. Component Verification: Dataset
    # ==========================================
    print("\n[3] Verifying Dataset and Transforms...")

    # Instantiate dataset
    dataset = AppleLeafDataset(
        train_subset, transforms=get_transforms("train"), mode="train"
    )

    # Check length
    assert len(dataset) == 20, "Dataset length mismatch."

    # Check item retrieval
    img, label = dataset[0]

    # Verify Image Tensor
    assert isinstance(img, torch.Tensor), "Image is not a tensor."
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Unexpected image shape: {img.shape}"

    # Verify Label
    assert isinstance(label, torch.Tensor), "Label is not a tensor."
    assert label.shape == (
        Config.NUM_CLASSES,
    ), f"Unexpected label shape: {label.shape}"

    print("    Dataset verification passed.")

    # ==========================================
    # 4. Component Verification: Model
    # ==========================================
    print("\n[4] Verifying Model Architecture...")

    model_name = "resnet34"
    model = get_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy input batch
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_input)

    # Verify output shape
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {output.shape}"

    print(f"    {model_name} instantiated and verified successfully.")
    del model, dummy_input
    torch.cuda.empty_cache()

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n[5] Running Training Loop (2 Folds)...")

    # We will run training for the defined number of folds (2)
    # run_training_fold saves the best model to Config.WORKING_DIR/models

    for fold in range(Config.N_FOLDS):
        print(f"    --- Fold {fold} ---")
        # In a real scenario, we would split based on folds.
        # Here we just reuse the subsets for demonstration.
        best_auc = run_training_fold(
            model_name="resnet34",
            train_df=train_subset,
            valid_df=val_subset,
            fold_idx=fold,
        )

        # Verify metric is valid
        assert isinstance(best_auc, float), "Returned metric is not a float."
        assert 0.0 <= best_auc <= 1.0, f"AUC score out of range: {best_auc}"

        # Verify model file creation
        expected_model_path = os.path.join(
            Config.WORKING_DIR, "models", f"resnet34_fold_{fold}.pth"
        )
        assert os.path.exists(
            expected_model_path
        ), f"Model file not found: {expected_model_path}"

    print("    Training loop completed successfully.")

    # ==========================================
    # 6. Inference Demonstration
    # ==========================================
    print("\n[6] Running Inference and Submission Generation...")

    # Create a temporary test metadata file for the demo
    # We'll use the validation subset but treat it as test data (no labels needed for input)
    demo_test_df = val_subset[["image_id", "file_path"]].copy()
    demo_test_path = os.path.join(Config.WORKING_DIR, "demo_test_metadata.csv")
    demo_test_df.to_csv(demo_test_path, index=False)

    # Monkey-patch Config.TEST_METADATA_PATH to point to our demo file
    # This allows generate_submission to pick up our small subset
    original_test_path = Config.TEST_METADATA_PATH
    Config.TEST_METADATA_PATH = demo_test_path

    try:
        # Generate submission
        # This will load models from Config.WORKING_DIR/models, run inference, and save to Config.SUBMISSION_PATH
        generate_submission()

        # Verify Submission File
        assert os.path.exists(
            Config.SUBMISSION_PATH
        ), "Submission file was not created."

        submission_df = pd.read_csv(Config.SUBMISSION_PATH)

        # Check shape: rows = num_test_samples, cols = image_id + num_classes
        expected_rows = len(demo_test_df)
        expected_cols = 1 + Config.NUM_CLASSES  # image_id + 4 classes

        assert submission_df.shape == (
            expected_rows,
            expected_cols,
        ), f"Submission shape mismatch. Expected ({expected_rows}, {expected_cols}), got {submission_df.shape}"

        # Check columns
        expected_columns = ["image_id"] + Config.CLASS_LABELS
        assert (
            list(submission_df.columns) == expected_columns
        ), "Submission columns mismatch."

        # Check values are probabilities
        pred_cols = Config.CLASS_LABELS
        assert (
            submission_df[pred_cols].values >= 0
        ).all(), "Negative probabilities found."
        assert (
            submission_df[pred_cols].values <= 1.0001
        ).all(), "Probabilities > 1 found."

        print("    Submission generated and verified successfully.")

    finally:
        # Restore original path (good practice)
        Config.TEST_METADATA_PATH = original_test_path

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
