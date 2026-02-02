import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd
import warnings

# Import library modules
from library import config, utils, dataset, models, engine, stacking


def run_demo():
    print("=== Starting Demonstration of Whale Detection Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Demo
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment...")

    # Set deterministic seeds
    utils.seed_everything(42)

    # Override config for speed and isolation
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 8
    config.WORK_DIR = "./working/demo_execution"
    config.CHECKPOINT_DIR = os.path.join(config.WORK_DIR, "checkpoints")
    config.CACHE_DIR = os.path.join(config.WORK_DIR, "cache")
    config.SUBMISSION_PATH = os.path.join(
        config.WORK_DIR, "submission/demo_submission.csv"
    )

    # Ensure directories exist
    if os.path.exists(config.WORK_DIR):
        shutil.rmtree(config.WORK_DIR)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    print(f"Working directory set to: {config.WORK_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 2] Loading Data (Debug Mode)...")

    # Use a small subset (50 samples) to speed up IO and processing
    debug_limit = 50
    dataloaders = dataset.get_dataloaders(
        load_cached_data=False, debug_limit=debug_limit
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    test_clips = dataloaders["test_clips"]

    # Assertions to verify data loading
    print("Verifying DataLoader shapes...")
    data_batch, target_batch = next(iter(train_loader))

    # Expected shape: (Batch, 1, F, T) -> (8, 1, 128, ~63) depending on padding
    assert data_batch.dim() == 4, f"Expected 4D input tensor, got {data_batch.dim()}"
    assert (
        data_batch.shape[0] == config.BATCH_SIZE
    ), f"Expected batch size {config.BATCH_SIZE}, got {data_batch.shape[0]}"
    assert data_batch.shape[1] == 1, f"Expected 1 channel, got {data_batch.shape[1]}"
    assert target_batch.shape[0] == config.BATCH_SIZE, "Target batch size mismatch"

    print(f"Batch Shape: {data_batch.shape}")
    print(f"Target Shape: {target_batch.shape}")
    print("Data loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 3] Initializing Model...")

    model_name = "tf_efficientnet_b0_ns"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.get_model(model_name, pretrained=False)  # False for speed in demo
    model = model.to(device)

    # Verify forward pass
    with torch.no_grad():
        dummy_input = data_batch.to(device)
        output = model(dummy_input)

    assert output.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Expected output shape {(config.BATCH_SIZE, 1)}, got {output.shape}"
    print("Model forward pass verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Training Loop (1 Epoch)...")

    # Run training for Fold 0
    # This calls train_one_epoch and validate internally
    best_auc, best_loss = engine.train_fold(
        fold_idx=0,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=1,
    )

    print(f"Training finished. Best AUC: {best_auc}, Best Loss: {best_loss}")

    # Verify checkpoints were created
    ckpt_auc = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_fold_0_auc.pth")
    ckpt_loss = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_fold_0_loss.pth")

    # Note: Depending on validation metrics, one or both might be saved.
    # Since we run 1 epoch, both 'best' conditions should trigger against initial values.
    if os.path.exists(ckpt_auc):
        print(f"Verified checkpoint exists: {ckpt_auc}")
    else:
        print(f"Warning: AUC Checkpoint not found (Validation AUC might be 0.0)")

    # -------------------------------------------------------------------------
    # 5. Stacking & Meta-Learner Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 5] Demonstrating Stacking/Meta-Learner...")

    # To demonstrate stacking without training 4 full models (2 archs * 2 metrics),
    # we simulate OOF predictions.

    # config.MODEL_ARCHITECTURES = ["tf_efficientnet_b0_ns", "resnet34"]
    # config.SAVE_METRICS = ["auc", "loss"]

    num_val_samples = len(val_loader.dataset)

    # Generate synthetic OOF predictions (probabilities)
    # In a real scenario, these come from engine.validate() on hold-out folds
    oof_preds = {}
    for arch in config.MODEL_ARCHITECTURES:
        for metric in config.SAVE_METRICS:
            key = f"{arch}_{metric}"
            # Random probabilities between 0 and 1
            oof_preds[key] = np.random.rand(num_val_samples)

    # Get actual targets from validation set
    val_targets = val_loader.dataset.targets

    # Train Meta Learner
    print("Training meta-learner on synthetic OOF data...")
    coef, intercept, meta_auc = stacking.train_meta_learner(oof_preds, val_targets)

    # Verify meta-learner artifacts
    assert os.path.exists(os.path.join(config.WORK_DIR, "meta_learner_coef.npy"))
    assert os.path.exists(os.path.join(config.WORK_DIR, "meta_learner_intercept.npy"))
    print("Meta-learner parameters saved successfully.")

    # Load Meta Learner (Verification)
    loaded_coef, loaded_intercept = stacking.load_meta_learner()
    np.testing.assert_array_almost_equal(coef, loaded_coef)
    print("Meta-learner loading verification passed.")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 6] Generating Submission...")

    # Simulate Test Predictions (Base Models)
    num_test_samples = len(test_loader.dataset)
    test_preds_base = {}
    for arch in config.MODEL_ARCHITECTURES:
        for metric in config.SAVE_METRICS:
            key = f"{arch}_{metric}"
            test_preds_base[key] = np.random.rand(num_test_samples)

    # Generate Stacked Predictions
    final_probs = stacking.predict_stack(test_preds_base, loaded_coef, loaded_intercept)

    assert len(final_probs) == num_test_samples, "Final probabilities length mismatch"

    # Create Submission File
    # Note: test_clips might be larger than debug_limit if not sliced correctly in get_dataloaders
    # dataset.get_dataloaders slices test_clips if debug_limit is set, so we are good.
    stacking.create_submission(
        final_probs, test_clips, output_path=config.SUBMISSION_PATH
    )

    # Verify Submission File
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    assert df_sub.shape == (
        num_test_samples,
        2,
    ), f"Submission shape mismatch: {df_sub.shape}"
    assert list(df_sub.columns) == [
        "clip",
        "probability",
    ], "Submission columns mismatch"

    print(f"Submission generated at {config.SUBMISSION_PATH}")
    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Suppress specific warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    try:
        run_demo()
    except Exception as e:
        print(f"\n!!! Demo Failed with Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
