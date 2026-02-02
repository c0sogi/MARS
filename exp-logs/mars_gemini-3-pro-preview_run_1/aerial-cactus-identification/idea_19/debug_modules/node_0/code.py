import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders
from library.models import CactusNeXt_RGB
from library.engine import train_one_epoch, validate_one_epoch, inference_tta
from library.stacking import get_data_vectors, train_meta_learner, predict_stacked


def run_pipeline_demo():
    print("=== Starting Cactus Classification Pipeline Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")
    # Override Config defaults to ensure the demo runs quickly
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Use a very small subset
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 2  # Moderate workers

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Loading DataLoaders...")
    # We use the Holistic model (RGB) for this demonstration
    train_loader, val_loader, test_loader = get_loaders(
        fold_idx=0,
        load_cached_data=False,  # Force processing to verify logic
        model_name=Config.MODEL_HOLISTIC,
    )

    # Verify Data Integrity
    print("    Verifying batch structure...")
    sample_batch = next(iter(train_loader))
    images, labels, fsize_norm, fsize_target = sample_batch

    # Check shapes: Images (B, 3, 32, 32), Labels (B), Meta (B, 1)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        32,
        32,
    ), f"Unexpected image shape: {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Unexpected label shape: {labels.shape}"
    assert fsize_norm.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Unexpected metadata shape: {fsize_norm.shape}"
    print("    Batch structure verified.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Model (CactusNeXt_RGB)...")
    device = Config.DEVICE
    model = CactusNeXt_RGB(num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Verify Forward Pass
    print("    Verifying forward pass...")
    with torch.no_grad():
        dummy_img = images.to(device)
        dummy_meta = fsize_norm.to(device)
        logits, quality = model(dummy_img, dummy_meta)

        assert logits.shape == (Config.BATCH_SIZE, 1), "Logits shape mismatch"
        assert quality.shape == (
            Config.BATCH_SIZE,
            1,
        ), "Quality prediction shape mismatch"
    print("    Forward pass verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\n[4] Starting Training Loop...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss, val_auc = validate_one_epoch(model, val_loader, device)

        print(
            f"    Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

    # -------------------------------------------------------------------------
    # 5. Inference
    # -------------------------------------------------------------------------
    print("\n[5] Running Inference (TTA)...")
    # Generate predictions for Validation set (to simulate OOF predictions for stacking)
    val_preds = inference_tta(model, val_loader, device)
    print(f"    Generated {len(val_preds)} validation predictions.")

    # Generate predictions for Test set
    test_preds = inference_tta(model, test_loader, device)
    print(f"    Generated {len(test_preds)} test predictions.")

    # -------------------------------------------------------------------------
    # 6. Stacking Ensemble Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Running Stacking Ensemble...")

    # Load metadata vectors for stacking
    # Note: get_data_vectors loads the full dataset metadata.
    # Since we are in DEBUG mode, our predictions (val_preds, test_preds) correspond
    # only to the subset used in the loaders. We must slice the vectors to match.

    val_ids, val_labels, val_fsizes = get_data_vectors(
        Config.VAL_METADATA_PATH, "val", load_cached_data=False
    )
    test_ids, test_labels, test_fsizes = get_data_vectors(
        Config.TEST_METADATA_PATH, "test", load_cached_data=False
    )

    # Align vectors with debug subset predictions
    val_limit = len(val_preds)
    test_limit = len(test_preds)

    val_ids_sub = val_ids[:val_limit]
    val_labels_sub = val_labels[:val_limit]
    val_fsizes_sub = val_fsizes[:val_limit]

    test_ids_sub = test_ids[:test_limit]
    test_fsizes_sub = test_fsizes[:test_limit]

    # Create dummy dictionary of predictions to simulate multiple models
    # In a real scenario, you would load predictions from different models (RepVGG, ResNet, NeXt)
    oof_preds_dict = {
        "NeXt_RGB": val_preds,
        "NeXt_RGB_Var": val_preds * 0.95 + 0.02,  # Simulated variation
    }

    test_preds_dict = {"NeXt_RGB": test_preds, "NeXt_RGB_Var": test_preds * 0.95 + 0.02}

    # Train Meta-Learner
    print("    Training Meta-Learner...")
    meta_model, meta_score = train_meta_learner(
        oof_preds_dict, val_labels_sub, val_fsizes_sub
    )
    print(f"    Meta-Learner Training AUC: {meta_score:.4f}")

    # Predict with Meta-Learner
    print("    Generating Final Submission...")
    submission_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    submission_df = predict_stacked(
        meta_model,
        test_preds_dict,
        test_fsizes_sub,
        test_ids_sub,
        output_path=submission_path,
    )

    # -------------------------------------------------------------------------
    # 7. Final Verification
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Output...")
    assert os.path.exists(submission_path), "Submission file was not created."
    assert len(submission_df) == len(test_preds), "Submission length mismatch."
    assert (
        "id" in submission_df.columns and "has_cactus" in submission_df.columns
    ), "Submission columns missing."

    print("    Submission Head:")
    print(submission_df.head())
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_pipeline_demo()
