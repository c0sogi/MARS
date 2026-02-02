import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_auc, AverageMeter
from library.data_loader import get_dataloaders
from library.model import SelfDistilledEfficientNet
from library.trainer import Trainer


def main():
    print("=== Starting Demonstration of Glioblastoma MGMT Prediction Pipeline ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Execution
    # --------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config parameters to ensure the script finishes quickly
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.DEBUG = True  # Enable debug mode logic
    Config.DEBUG_SAMPLE_SIZE = 12  # Use only 12 samples (enough for a few batches)
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure working directory is clean for this run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("    Configuration updated: 1 Epoch, Debug Mode enabled.\n")

    # --------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # --------------------------------------------------------------------------
    print("[2] Verifying Utility Functions...")

    # Test AUC Calculation
    y_true = [0, 0, 1, 1]
    y_pred_perfect = [0.1, 0.2, 0.8, 0.9]
    y_pred_bad = [0.9, 0.8, 0.2, 0.1]

    auc_perfect = calculate_auc(y_true, y_pred_perfect)
    auc_bad = calculate_auc(y_true, y_pred_bad)

    assert auc_perfect == 1.0, f"Expected AUC 1.0, got {auc_perfect}"
    assert auc_bad == 0.0, f"Expected AUC 0.0, got {auc_bad}"
    print("    AUC calculation logic verified.")

    # Test AverageMeter
    meter = AverageMeter("Test")
    meter.update(val=10, n=2)  # sum=20, count=2
    meter.update(val=20, n=2)  # sum=60, count=4
    assert meter.avg == 15.0, f"Expected AverageMeter avg 15.0, got {meter.avg}"
    print("    AverageMeter logic verified.\n")

    # --------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("[3] Verifying Model Architecture...")

    model = SelfDistilledEfficientNet()
    model.eval()

    # Create a dummy input tensor: (Batch=2, Channels=12, Height=256, Width=256)
    # Channels=12 corresponds to 4 modalities * 3 slices
    dummy_input = torch.randn(
        2, Config.TOTAL_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE
    )

    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: Should be (Batch, 1) for binary classification logits
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print(f"    Model forward pass successful. Output shape: {output.shape}")

    # Verify the first layer modification (Grouped Conv stem)
    first_layer = model.backbone.features[0][0]
    assert first_layer.in_channels == 12, "First layer in_channels should be 12"
    assert (
        first_layer.groups == 4
    ), "First layer should use grouped convolutions (groups=4)"
    print("    Model stem modification verified.\n")

    # --------------------------------------------------------------------------
    # 4. Verify Data Loading Pipeline
    # --------------------------------------------------------------------------
    print("[4] Verifying Data Loading Pipeline...")

    # Load metadata dataframes
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    print(
        f"    Loaded metadata. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # Initialize Dataloaders (this triggers the caching mechanism in debug mode)
    print("    Initializing dataloaders (calculating peak intensities)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, debug=True
    )

    # Verify Train Loader Batch
    # Train loader returns: view_a, view_b, targets
    view_a, view_b, targets = next(iter(train_loader))

    assert view_a.shape == (
        Config.BATCH_SIZE,
        12,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Train View A shape mismatch"
    assert view_b.shape == (
        Config.BATCH_SIZE,
        12,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Train View B shape mismatch"
    assert targets.shape == (Config.BATCH_SIZE,), "Train Targets shape mismatch"
    print("    Train loader batch shapes verified.")

    # Verify Test Loader Batch (Test Time Augmentation structure)
    # Test loader returns: tta_tensor, subject_ids
    tta_tensor, subject_ids = next(iter(test_loader))

    # Expected shape: (Batch, 3_views, 12_channels, H, W)
    assert len(tta_tensor.shape) == 5, "Test TTA tensor should be 5D"
    assert tta_tensor.shape[1] == 3, "Test tensor should have 3 TTA views"
    assert tta_tensor.shape[2] == 12, "Test tensor should have 12 channels"
    print("    Test loader TTA batch shapes verified.\n")

    # --------------------------------------------------------------------------
    # 5. Verify Training & Inference Loop
    # --------------------------------------------------------------------------
    print("[5] Verifying Full Training Cycle...")

    trainer = Trainer(device=Config.DEVICE)

    # Run the training loop (Debug mode uses the subsets created above)
    # This will train for 1 epoch, validate, save checkpoint, and run inference
    trainer.run_training(debug=True)

    # Verify outputs exist
    assert os.path.exists(Config.CHECKPOINT_PATH), "Model checkpoint was not created."
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content
    submission = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission file created with {len(submission)} rows.")
    assert "BraTS21ID" in submission.columns, "Submission missing BraTS21ID column"
    assert "MGMT_value" in submission.columns, "Submission missing MGMT_value column"

    # Check if predictions are probabilities (0-1)
    preds = submission["MGMT_value"].values
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions outside [0, 1] range"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
