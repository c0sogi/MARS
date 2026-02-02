import os
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import from provided libraries
from library.utils import set_seed
from library.dataset import SaltDataset
from library.model import WideLinkNetResNet34
from library.losses import CombinedLoss
from library.engine import Trainer
from library.inference import predict_proba, optimize_threshold, generate_submission_csv

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("--- Starting Salt Segmentation Demo ---")

    # 1. Configuration and Setup
    # --------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    BATCH_SIZE = 8
    EPOCHS = 2
    # Use a small subset for speed to demonstrate functionality within time limits
    SUBSET_SIZE_TRAIN = 32
    SUBSET_SIZE_VAL = 16
    SUBSET_SIZE_TEST = 16

    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo_run"

    set_seed(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Working Directory: {WORKING_DIR}")

    # 2. Data Preparation
    # -------------------
    print("\n[Step 1] Loading and subsetting metadata...")
    # Load metadata
    train_df_full = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df_full = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df_full = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Create subsets
    train_df = train_df_full.head(SUBSET_SIZE_TRAIN).copy()
    val_df = val_df_full.head(SUBSET_SIZE_VAL).copy()
    test_df = test_df_full.head(SUBSET_SIZE_TEST).copy()

    print(
        f"Subset sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # Instantiate Datasets
    # We set load_cached_data=False to prevent loading any existing full-dataset cache
    # and ensure we only use our small subset for this demo.
    print("Instantiating SaltDataset objects...")
    train_dataset = SaltDataset(mode="train", df=train_df, load_cached_data=False)
    val_dataset = SaltDataset(mode="val", df=val_df, load_cached_data=False)
    test_dataset = SaltDataset(mode="test", df=test_df, load_cached_data=False)

    # Verification: Check item shapes
    # Dataset returns: img (1, 128, 128), mask (1, 128, 128), depth (1,), id
    sample_img, sample_mask, sample_depth, sample_id = train_dataset[0]

    assert sample_img.shape == (
        1,
        128,
        128,
    ), f"Unexpected image shape: {sample_img.shape}"
    assert sample_mask.shape == (
        1,
        128,
        128,
    ), f"Unexpected mask shape: {sample_mask.shape}"
    assert sample_depth.shape == (1,), f"Unexpected depth shape: {sample_depth.shape}"
    print("Dataset shapes verified.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 3. Model Initialization
    # -----------------------
    print("\n[Step 2] Initializing Model...")
    model = WideLinkNetResNet34().to(DEVICE)

    # Verification: Forward pass with dummy data
    dummy_img = torch.randn(2, 1, 128, 128).to(DEVICE)
    dummy_depth = torch.randn(2, 1).to(DEVICE)
    with torch.no_grad():
        dummy_out = model(dummy_img, dummy_depth)

    assert dummy_out.shape == (
        2,
        1,
        128,
        128,
    ), f"Model output shape mismatch: {dummy_out.shape}"
    print("Model forward pass successful.")

    # 4. Training Setup
    # -----------------
    print("\n[Step 3] Setting up Training Components...")
    # Combined Loss: BCE + Lovasz
    criterion = CombinedLoss(bce_weight=1.0, lovasz_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        patience=5,
        save_dir=WORKING_DIR,
    )

    # 5. Training Execution
    # ---------------------
    print("\n[Step 4] Starting Training Loop...")
    # We pass None for test_loader to skip the automatic submission generation in Trainer,
    # so we can demonstrate the inference module manually later.
    trainer.fit(train_loader, val_loader, test_loader=None, epochs=EPOCHS)

    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Best model checkpoint was not saved.")
    print(f"Training complete. Best model saved to {best_model_path}")

    # 6. Inference and Evaluation
    # ---------------------------
    print("\n[Step 5] Running Inference and Threshold Optimization...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    # Predict on validation set using library.inference.predict_proba
    # This function handles TTA and unpadding (cropping back to 101x101)
    val_results = predict_proba(model, val_loader, DEVICE, force_zero_depth=False)
    val_preds = val_results["predictions"]  # Shape: (N, 101, 101)
    val_masks = val_results["masks"]  # Shape: (N, 101, 101)

    assert val_preds.shape == (
        SUBSET_SIZE_VAL,
        101,
        101,
    ), f"Validation prediction shape mismatch: {val_preds.shape}"

    # Optimize threshold using library.inference.optimize_threshold
    best_thresh = optimize_threshold(val_masks, val_preds)
    print(f"Optimal Threshold: {best_thresh:.4f}")

    # 7. Submission Generation
    # ------------------------
    print("\n[Step 6] Generating Submission for Test Set...")

    # Predict on test set
    # force_zero_depth=True is often used for test inference to be robust against depth distribution shifts
    test_results = predict_proba(model, test_loader, DEVICE, force_zero_depth=True)
    test_ids = test_results["ids"]
    test_preds = test_results["predictions"]

    submission_path = os.path.join(WORKING_DIR, "submission.csv")

    # Generate CSV using library.inference.generate_submission_csv
    generate_submission_csv(
        test_ids, test_preds, best_thresh, output_path=submission_path
    )

    # Verification
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not created.")

    submission_df = pd.read_csv(submission_path)
    print("Submission Head:")
    print(submission_df.head())

    assert len(submission_df) == SUBSET_SIZE_TEST, "Submission row count mismatch."
    assert (
        "id" in submission_df.columns and "rle_mask" in submission_df.columns
    ), "Submission columns mismatch."

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
