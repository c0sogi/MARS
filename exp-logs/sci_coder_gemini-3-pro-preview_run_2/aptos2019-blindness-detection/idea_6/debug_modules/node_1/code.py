import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.data import get_dataloaders, RetinopathyDataset
from library.model import DRModel
from library.engine import run_fold
from library.inference import predict_ensemble


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # -------------------------------------------------------------------------
    print(">>> Setting up configuration for demo execution...")

    # Set a specific working directory for this demo to avoid conflicts
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override Config for speed and debugging
    # DEBUG=True forces the data loader to take only a small head of the dataframes
    Config.override(
        DEBUG=True,
        EPOCHS=1,  # Run only 1 epoch
        NUM_FOLDS=1,  # Run only 1 fold
        WORKING_DIR=demo_working_dir,
        SUBMISSION_PATH=os.path.join(demo_working_dir, "submission.csv"),
        NUM_WORKERS=2,  # Reduce workers for the small demo
        BATCH_SIZE=4,  # Small batch size
        SWA_START_EPOCH=0,  # Trigger SWA immediately to test that code path
        USE_SWA=True,
        MODEL_CNN={
            "name": "tf_efficientnet_b5_ns",
            "img_size": 256,  # Reduce size for speed in demo
            "batch_size": 4,
            "dropout": 0.2,
            "checkpoint_prefix": "effnet_b5",
        },
    )

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Data Pipeline...")

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        img_size=Config.MODEL_CNN["img_size"],
        batch_size=Config.MODEL_CNN["batch_size"],
        load_cached_data=False,  # Force reload from CSV since we changed DEBUG flag
    )

    # Verify Train Loader
    images, targets = next(iter(train_loader))
    print(f"Train Batch - Images: {images.shape}, Targets: {targets.shape}")

    # Assertions for shapes
    # Images: (B, 3, H, W)
    assert images.dim() == 4, "Images should be 4D tensor (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert images.shape[2] == Config.MODEL_CNN["img_size"], "Height mismatch"
    assert images.shape[3] == Config.MODEL_CNN["img_size"], "Width mismatch"
    # Targets: (B,) or (B, 1) depending on collate, usually (B) from dataset
    # The dataset returns scalar tensors, loader stacks them to (B)
    assert targets.dim() == 1, "Targets should be 1D tensor"
    assert len(targets) == Config.MODEL_CNN["batch_size"], "Batch size mismatch"

    print("Data Pipeline verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Model Instantiation...")

    # Instantiate model with pretrained=False for speed (avoid download if possible)
    # Note: In real training, we use pretrained=True.
    model = DRModel(model_name=Config.MODEL_CNN["name"], pretrained=False)
    model.eval()

    # Dummy forward pass
    with torch.no_grad():
        output = model(images)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        Config.MODEL_CNN["batch_size"],
        1,
    ), "Output shape should be (B, 1)"

    del model
    print("Model verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution (Fold 0)
    # -------------------------------------------------------------------------
    print("\n>>> Executing Training Loop (Fold 0)...")

    # We run the engine's run_fold function
    # This handles Model init, Optimizer, Scheduler, SWA, Training, Validation, Saving
    run_fold(
        fold=0,
        model_config=Config.MODEL_CNN,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Verify outputs exist
    best_model_path = os.path.join(Config.WORKING_DIR, "effnet_b5_fold_0_best.pth")
    swa_model_path = os.path.join(Config.WORKING_DIR, "effnet_b5_fold_0_swa.pth")

    if os.path.exists(best_model_path):
        print(f"Checkpoint found: {best_model_path}")
    else:
        raise FileNotFoundError(f"Expected checkpoint not found: {best_model_path}")

    if os.path.exists(swa_model_path):
        print(f"SWA Checkpoint found: {swa_model_path}")
    else:
        # SWA might not save if validation didn't improve or logic didn't trigger,
        # but with SWA_START_EPOCH=0 and 1 epoch, it should trigger "Finalizing SWA".
        print(
            "Warning: SWA checkpoint not found (might be expected if logic didn't trigger)."
        )

    # -------------------------------------------------------------------------
    # 5. Inference Execution
    # -------------------------------------------------------------------------
    print("\n>>> Executing Inference Pipeline...")

    # This will look for checkpoints in Config.WORKING_DIR and generate submission.csv
    # It iterates over Config.NUM_FOLDS (which we set to 1)
    predict_ensemble(load_cached_data=True)

    # Verify submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(df_sub.head())

    # Verify submission format
    assert "id_code" in df_sub.columns
    assert "diagnosis" in df_sub.columns
    # In DEBUG mode, test set is 20 rows
    expected_len = 20
    assert (
        len(df_sub) == expected_len
    ), f"Submission length mismatch. Expected {expected_len}, got {len(df_sub)}"

    print("Inference pipeline verified successfully.")

    # -------------------------------------------------------------------------
    # 6. Metric Logic Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Metric Logic (QWK)...")

    # Case 1: Perfect agreement
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    score = quadratic_weighted_kappa(y_true, y_pred)
    print(f"Perfect Agreement Score: {score}")
    assert np.isclose(score, 1.0), "QWK should be 1.0 for perfect agreement"

    # Case 2: Complete disagreement
    y_true_bad = np.array([0, 0, 0, 0, 0])
    y_pred_bad = np.array([4, 4, 4, 4, 4])
    score_bad = quadratic_weighted_kappa(y_true_bad, y_pred_bad)
    print(f"Disagreement Score: {score_bad}")
    # Score should be 0 or negative (random/worse than random)
    assert score_bad <= 0.0, "QWK should be <= 0 for poor agreement"

    print("Metric logic verified successfully.")

    print("\n>>> Demo Execution Complete. All systems operational.")


if __name__ == "__main__":
    main()
