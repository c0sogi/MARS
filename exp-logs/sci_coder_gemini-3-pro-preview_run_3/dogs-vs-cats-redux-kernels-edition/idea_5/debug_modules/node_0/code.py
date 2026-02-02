import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library import utils, dataset, models, engine, stacking


def run_demo():
    print("=== Starting Dog vs Cat Solution Demo ===")

    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("\n[1] Setting up configuration and seeding...")

    # Set seed for reproducibility
    utils.seed_everything(Config.SEED)

    # Override Config for speed in this demo
    Config.DEBUG = True
    Config.EPOCHS = 1
    # Use a smaller batch size for the demo to ensure it runs on any GPU/CPU quickly
    Config.BATCH_SIZE = 16

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Device: {Config.DEVICE}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Loading...")

    # Get dataloaders for Fold 0 in debug mode
    train_loader, val_loader = dataset.get_dataloaders(
        fold_idx=0, debug=True, batch_size=Config.BATCH_SIZE
    )

    # Verify Train Loader
    images, targets = next(iter(train_loader))
    print(f"    Train Batch Shape: {images.shape}")
    print(f"    Targets Shape: {targets.shape}")

    # Assertions to ensure data pipeline is correct
    assert len(train_loader) > 0, "Train loader is empty"
    assert images.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert images.shape[2] == Config.IMG_SIZE, "Image height mismatch"
    assert images.shape[3] == Config.IMG_SIZE, "Image width mismatch"

    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Model Initialization...")

    # Initialize the first model from config (ResNet101)
    model_name = Config.MODEL_CONFIGS[0]["name"]
    print(f"    Initializing model: {model_name}")

    # We use pretrained=False here just to speed up the demo (avoiding large download)
    # In a real run, keep pretrained=True as per Config default.
    # However, to strictly follow the prompt's environment where packages are installed,
    # we'll try to use the default. If it downloads, it downloads.
    model = models.get_model(model_name, pretrained=True, num_classes=1)
    model.to(Config.DEVICE)

    # 4. Training and Validation (Engine)
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Training and Validation Loop...")

    # Setup Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Setup Mixed Precision Scaler
    # Note: torch.amp.GradScaler('cuda') is specific to CUDA.
    # If CPU, enabled=False handles it, but the init arg 'device' is 'cuda' by default in recent torch.
    scaler = torch.amp.GradScaler("cuda", enabled=(Config.DEVICE == "cuda"))

    # Train for one epoch
    print("    Training for 1 epoch (Debug subset)...")
    train_loss = engine.train_one_epoch(
        model, optimizer, train_loader, Config.DEVICE, scaler
    )
    print(f"    Train Loss: {train_loss:.4f}")

    # Validate
    print("    Validating...")
    val_loss, val_acc = engine.validate(model, val_loader, Config.DEVICE)
    print(f"    Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # Assertions
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert 0.0 <= val_acc <= 1.0, "Validation accuracy out of bounds"

    # Save a dummy checkpoint to demonstrate utility
    checkpoint_path = os.path.join(Config.WORKING_DIR, "demo_checkpoint.pth")
    utils.save_checkpoint(
        {
            "epoch": 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path,
        is_best=True,
    )
    assert os.path.exists(checkpoint_path), "Checkpoint failed to save"

    # 5. Stacking / Meta-Learner
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Stacking/Meta-Learner...")

    # Simulate OOF (Out-Of-Fold) predictions for 2 models
    # In a real scenario, these come from saving val predictions across 5 folds.
    n_samples = 100
    np.random.seed(Config.SEED)

    # Ground truth labels (0 or 1)
    y_true = np.random.randint(0, 2, n_samples)

    # Simulate predictions from Model A (ResNet) - slightly correlated with truth
    pred_a = np.clip(y_true * 0.8 + np.random.rand(n_samples) * 0.2, 0.01, 0.99)

    # Simulate predictions from Model B (ConvNeXt) - slightly correlated
    pred_b = np.clip(y_true * 0.7 + np.random.rand(n_samples) * 0.3, 0.01, 0.99)

    oof_df = pd.DataFrame(
        {"label": y_true, "resnet_pred": pred_a, "convnext_pred": pred_b}
    )

    feature_cols = ["resnet_pred", "convnext_pred"]

    # Train Meta Learner
    meta_model, meta_loss = stacking.train_meta_learner(
        oof_df, feature_cols, target_col="label"
    )

    assert meta_model is not None, "Meta learner failed to train"
    print(f"    Meta-Learner Log Loss: {meta_loss:.4f}")

    # 6. Prediction and Submission
    # -------------------------------------------------------------------------
    print("\n[6] Demonstrating Prediction and Submission...")

    # Simulate Test Data predictions from Level-1 models
    n_test = 50
    test_ids = np.arange(1, n_test + 1)

    test_pred_a = np.random.rand(n_test)
    test_pred_b = np.random.rand(n_test)

    test_df = pd.DataFrame({"resnet_pred": test_pred_a, "convnext_pred": test_pred_b})

    # Predict using Meta Learner
    final_probs = stacking.predict_meta_learner(test_df, feature_cols, meta_model)

    assert len(final_probs) == n_test, "Prediction length mismatch"
    assert np.all(
        (final_probs >= 0) & (final_probs <= 1)
    ), "Probabilities out of bounds"

    # Generate Submission File
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    stacking.generate_submission(test_ids, final_probs, output_path=submission_path)

    assert os.path.exists(submission_path), "Submission file not created"

    # Verify file content
    sub_df = pd.read_csv(submission_path)
    print(f"    Submission file created with {len(sub_df)} rows.")
    print(f"    Head:\n{sub_df.head(3)}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
