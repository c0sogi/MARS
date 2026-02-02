import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path to import library modules correctly
sys.path.append(os.getcwd())

# Import provided library modules
import library.config as cfg
import library.utils as utils
import library.data as data
import library.model as model_lib
import library.engine as engine


def main():
    print("=== Starting Retinopathy Classifier Demonstration ===")

    # 1. Configuration and Setup
    # Override configuration for a fast demonstration
    print("Configuring parameters for speed...")
    cfg.DEBUG = True
    cfg.DEBUG_SAMPLE_SIZE = 50  # Use a tiny subset of data
    cfg.IMG_SIZE = 224  # Smaller image size for faster processing
    cfg.BATCH_SIZE = 8  # Small batch size
    cfg.EPOCHS = 2  # Minimal epochs to prove the loop works
    cfg.MODEL_NAME = "tf_efficientnet_b0_ns"  # Use a smaller backbone
    cfg.NUM_WORKERS = 2  # Reduce worker overhead

    # Set up a specific working directory for this demo
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)
    cfg.WORKING_DIR = demo_dir

    model_save_path = os.path.join(demo_dir, "model_best.pth")
    submission_path = os.path.join(demo_dir, "submission.csv")

    # Set seeds
    utils.seed_everything(cfg.SEED)

    # 2. Data Loading
    print("\n=== Testing Data Loading ===")
    # Force reload to test CSV reading logic instead of loading cached parquet if it exists from previous runs
    train_loader, val_loader, test_loader = data.get_loaders(load_cached_data=False)

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches:   {len(val_loader)}")
    print(f"Test Loader Batches:  {len(test_loader)}")

    # Verify batch structure
    images, labels = next(iter(train_loader))
    print(f"Sample Batch - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions for data integrity
    assert images.shape == (
        cfg.BATCH_SIZE,
        3,
        cfg.IMG_SIZE,
        cfg.IMG_SIZE,
    ), f"Incorrect image shape: {images.shape}"
    assert labels.shape == (cfg.BATCH_SIZE,), f"Incorrect label shape: {labels.shape}"
    assert labels.dtype == torch.float, "Labels must be float for regression loss."

    # 3. Model Initialization
    print("\n=== Testing Model Initialization ===")
    device = cfg.DEVICE
    print(f"Using device: {device}")

    model = model_lib.RetinopathyModel(pretrained=True)
    model.to(device)

    # Verify forward pass with dummy data
    dummy_input = torch.randn(2, 3, cfg.IMG_SIZE, cfg.IMG_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    # 4. Training Loop
    print("\n=== Testing Training Loop ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR
    )

    best_score = engine.run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=cfg.EPOCHS,
        patience=cfg.PATIENCE,
        save_path=model_save_path,
    )

    print(f"Training finished. Best QWK Score: {best_score}")
    assert os.path.exists(model_save_path), "Model checkpoint was not saved."

    # 5. Inference and Submission
    print("\n=== Testing Inference and Submission ===")
    # Load the saved model weights
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    engine.make_submission(
        model=model, test_loader=test_loader, device=device, save_path=submission_path
    )

    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify submission file format
    df_sub = pd.read_csv(submission_path)
    print("Submission File Head:")
    print(df_sub.head())

    assert list(df_sub.columns) == [
        "id_code",
        "diagnosis",
    ], "Submission columns are incorrect."
    assert len(df_sub) > 0, "Submission file is empty."
    # Check if diagnosis values are integers (as required by the metric/submission format)
    assert pd.api.types.is_integer_dtype(
        df_sub["diagnosis"]
    ), "Diagnosis column must be integer."

    # 6. Metric Utility Verification
    print("\n=== Testing Metric Utility ===")
    # Create synthetic ground truth and predictions
    # Case: Perfect prediction (after rounding)
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred_raw = np.array([0.2, 1.1, 1.9, 3.1, 3.8])  # Should round to 0, 1, 2, 3, 4

    score = utils.compute_score(y_true, y_pred_raw)
    print(f"Calculated QWK Score (Perfect Match): {score}")

    assert score > 0.99, "Metric calculation failed for perfect predictions."

    # Case: Complete mismatch
    y_true_mismatch = np.array([0, 0, 0])
    y_pred_mismatch = np.array([4.0, 4.0, 4.0])
    score_mismatch = utils.compute_score(y_true_mismatch, y_pred_mismatch)
    print(f"Calculated QWK Score (Mismatch): {score_mismatch}")

    assert score_mismatch < 0.1, "Metric calculation failed for mismatched predictions."

    print("\n=== All Demonstrations Passed Successfully ===")


if __name__ == "__main__":
    main()
