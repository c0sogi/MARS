import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.data import prepare_folds, get_loaders, get_test_loader
from library.model import get_model
from library.loss import AsymmetricLoss
from library.trainer import run_training, Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== Starting Library Demonstration ====")

    # 1. Setup Configuration
    # We use debug=True to reduce data size and epochs for a quick run.
    # We override epochs to 1 for this demonstration to be very fast.
    cfg = Config(debug=True, epochs=1, batch_size=4)
    # Ensure the working directory is specific for this demo to avoid conflicts
    cfg.working_dir = "./working/demo_execution"
    os.makedirs(cfg.working_dir, exist_ok=True)

    print(f"Configuration initialized. Working dir: {cfg.working_dir}")
    print(f"Debug Mode: {cfg.debug}, Device: {cfg.device}")

    # Set seeds for reproducibility
    seed_everything(cfg.seed)

    # 2. Data Preparation
    print("\n[Step 1] Preparing Data Folds...")
    # This will read metadata, apply iterative stratification, and save to parquet
    df_folds = prepare_folds(cfg, load_cached_data=False)

    # Validate Folds DataFrame
    assert "fold" in df_folds.columns, "Folds DataFrame missing 'fold' column"
    assert len(df_folds) > 0, "Folds DataFrame is empty"
    print(f"Folds generated. Shape: {df_folds.shape}")

    print("\n[Step 2] Creating DataLoaders (Fold 0)...")
    train_loader, val_loader = get_loaders(fold=0, df=df_folds, cfg=cfg)

    # Validate Train Loader
    images, labels = next(iter(train_loader))
    print(f"Batch Shapes -> Images: {images.shape}, Labels: {labels.shape}")

    # Assertions for data shapes
    # Expected: (Batch, 3, 224, 224) for images, (Batch, 19) for labels
    assert images.shape == (
        cfg.batch_size,
        3,
        224,
        224,
    ), f"Incorrect image shape: {images.shape}"
    assert labels.shape == (
        cfg.batch_size,
        cfg.num_classes,
    ), f"Incorrect label shape: {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.float32, "Labels should be float32"

    print("\n[Step 3] Creating Test Loader...")
    test_loader = get_test_loader(cfg)
    test_images, test_ids = next(iter(test_loader))
    assert test_images.shape[1:] == (3, 224, 224), "Incorrect test image dimensions"
    print("Test loader created successfully.")

    # 3. Model Initialization
    print("\n[Step 4] Initializing Model (resnet18)...")
    model = get_model(cfg, model_name="resnet18")

    # Move to CPU for simple shape verification (Trainer will move to GPU)
    model.to("cpu")
    model.eval()

    # Validate Forward Pass
    with torch.no_grad():
        # Use the batch fetched from train_loader (move to cpu)
        output = model(images.cpu())

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        cfg.batch_size,
        cfg.num_classes,
    ), "Model output shape mismatch"

    # 4. Loss Function
    print("\n[Step 5] Testing Loss Function...")
    criterion = AsymmetricLoss()
    loss = criterion(output, labels.cpu())

    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # 5. Training Loop
    print("\n[Step 6] Running Training Loop (1 Epoch)...")
    # Re-initialize model to ensure clean state and let Trainer handle device placement
    model = get_model(cfg, model_name="resnet18")

    # Instantiate Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        fold=0,
        model_name="resnet18",
    )

    # Run fitting
    best_auc = trainer.fit()

    print(f"Training completed. Best AUC: {best_auc:.4f}")

    # Validate Checkpoint Creation
    checkpoint_path = os.path.join(cfg.working_dir, "resnet18_fold_0_best.pth")
    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint file not found at {checkpoint_path}"
    print(f"Checkpoint verified at: {checkpoint_path}")

    # 6. Metric Calculation
    print("\n[Step 7] Verifying Metric Calculation...")
    # Create synthetic ground truth and predictions
    # Case: 2 samples, 19 classes
    y_true = np.zeros((2, 19))
    y_true[0, 0] = 1  # Class 0 present in sample 0
    y_true[1, 1] = 1  # Class 1 present in sample 1

    y_pred = np.zeros((2, 19))
    y_pred[0, 0] = 0.9  # Good prediction
    y_pred[1, 1] = 0.8  # Good prediction
    y_pred[0, 1] = 0.1  # Good negative prediction

    # Calculate AUC
    # Note: calculate_roc_auc handles cases where a class has only 1 label type in batch by skipping
    # In this tiny synthetic batch, many classes are all-zeros, so they are skipped.
    auc_score = calculate_roc_auc(y_true, y_pred)
    print(f"Synthetic AUC Score: {auc_score:.4f}")

    assert 0 <= auc_score <= 1, "AUC score out of range"

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
