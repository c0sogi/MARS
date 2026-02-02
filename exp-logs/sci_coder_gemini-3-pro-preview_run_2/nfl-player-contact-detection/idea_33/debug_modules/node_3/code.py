import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import the provided library modules
from library import config, utils, data_processor, model, trainer


def run_demo():
    print("=== Starting Library Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Isolation
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set a separate working directory for this demo to avoid conflicts
    DEMO_WORKING_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Patch the config module directly
    config.WORKING_DIR = DEMO_WORKING_DIR
    config.EPOCHS = 1  # Train for only 1 epoch
    config.DEBUG_SAMPLE_SIZE = 200  # Use only 200 samples for processing
    config.BATCH_SIZE = 32  # Small batch size
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure reproducibility
    utils.seed_everything(seed=42)
    print("Configuration patched: 1 Epoch, 200 Samples, Batch Size 32.")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test Shortest Arc
    # Arc between 10 and 350 should be 20, not 340
    angle1 = np.array([10.0, 10.0])
    angle2 = np.array([350.0, 20.0])
    arcs = utils.calculate_shortest_arc(angle1, angle2)
    expected_arcs = np.array([20.0, 10.0])
    np.testing.assert_allclose(
        arcs, expected_arcs, err_msg="Shortest arc calculation failed"
    )
    print(" - utils.calculate_shortest_arc: OK")

    # Test Focal Loss
    focal_loss = utils.FocalLoss(reduction="mean")
    logits = torch.tensor([10.0, -10.0], requires_grad=True)  # Preds: ~1, ~0
    targets = torch.tensor([1.0, 0.0])  # True: 1, 0
    loss = focal_loss(logits, targets)
    loss.backward()

    assert loss.item() < 0.1, "Focal loss should be low for correct predictions"
    assert logits.grad is not None, "Gradients not flowing through FocalLoss"
    print(" - utils.FocalLoss: OK")

    # -------------------------------------------------------------------------
    # 3. Verify Data Processing
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Processor...")

    # We force reload to demonstrate processing logic
    train_loader, val_loader, test_loader, df_test = data_processor.prepare_datasets(
        load_cached_data=False, debug_sample=config.DEBUG_SAMPLE_SIZE
    )

    # Check Loaders
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Validation loader is empty"

    # Check Batch Shape
    features, targets = next(iter(train_loader))
    print(f" - Train Batch Shape: Features {features.shape}, Targets {targets.shape}")
    assert features.dim() == 2, "Features should be 2D (Batch, Features)"
    assert targets.dim() == 1, "Targets should be 1D (Batch)"

    # Check Feature Columns extraction
    # We need the schema to initialize the model later
    train_parquet_path = os.path.join(config.WORKING_DIR, "train_features.parquet")
    df_schema = pd.read_parquet(train_parquet_path).head(0)
    feature_names = data_processor.get_feature_columns(df_schema.columns)
    print(f" - Extracted {len(feature_names)} feature names.")

    # Verify we have both kinematic and visual features
    has_vis = any("area" in f for f in feature_names)
    has_kin = any("speed" in f for f in feature_names)
    assert has_vis and has_kin, "Features must contain both visual and kinematic data"
    print(" - Data Processing: OK")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture (IP-RVN)...")

    device = torch.device("cpu")  # Use CPU for simple logic check
    net = model.IPRVN(feature_names).to(device)

    # Check Stream Splitting
    print(f" - Kinematic Features: {len(net.kin_idx)}")
    print(f" - Visual Features: {len(net.vis_idx)}")
    assert len(net.kin_idx) + len(net.vis_idx) == len(
        feature_names
    ), "Feature splitting mismatch"

    # Test Input Clamping Layer
    # Create input with outlier values
    dummy_input = torch.randn(2, len(feature_names)).to(device)
    dummy_input[0, 0] = 1000.0  # Should be clamped to 50
    dummy_input[0, 1] = -1000.0  # Should be clamped to -50

    # Access the clamping layer directly to verify
    clamped = net.clamping(dummy_input)
    assert clamped.max() <= 50.0, "InputClampingLayer failed max clamp"
    assert clamped.min() >= -50.0, "InputClampingLayer failed min clamp"
    print(" - Input Clamping: OK")

    # Test Forward Pass
    logits = net(dummy_input)
    assert logits.shape == (
        2,
        1,
    ), f"Output shape mismatch. Expected (2, 1), got {logits.shape}"
    print(" - Forward Pass: OK")

    # -------------------------------------------------------------------------
    # 5. Verify Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Training Process...")

    # We use the provided trainer.train_model which handles the loop
    # It uses the config values we patched earlier
    try:
        trainer.train_model(debug_sample=config.DEBUG_SAMPLE_SIZE)
    except Exception as e:
        raise AssertionError(f"Training loop failed: {e}")

    # Check if artifacts were created
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    thresh_path = os.path.join(config.WORKING_DIR, "best_threshold.npy")

    assert os.path.exists(model_path), "Model file was not saved"
    assert os.path.exists(thresh_path), "Threshold file was not saved"
    print(" - Training Loop: OK (Model and Threshold saved)")

    # -------------------------------------------------------------------------
    # 6. Verify Inference
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Inference Process...")

    # Override submission path to working dir for demo
    config.SUBMISSION_DIR = DEMO_WORKING_DIR
    config.SUBMISSION_PATH = os.path.join(DEMO_WORKING_DIR, "submission.csv")

    try:
        trainer.predict()
    except Exception as e:
        raise AssertionError(f"Inference failed: {e}")

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created"

    # Validate submission format
    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    assert "contact_id" in sub_df.columns, "Submission missing contact_id"
    assert "contact" in sub_df.columns, "Submission missing contact column"
    assert not sub_df.empty, "Submission file is empty"
    print(f" - Submission generated with {len(sub_df)} rows.")
    print(" - Inference: OK")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
