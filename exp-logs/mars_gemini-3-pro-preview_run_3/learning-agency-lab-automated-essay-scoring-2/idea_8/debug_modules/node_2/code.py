import os
import sys
import torch
import pandas as pd
import numpy as np

# 1. Setup Config FIRST to ensure downstream modules pick up changes
from library.config import Config

# --- Configuration Overrides for Fast Demo ---
print("Configuring environment for rapid demonstration...")
Config.DEBUG = True  # Triggers data subsampling (100 train, 50 val, 20 test samples)
Config.EXP_NAME = "demo_run"

# Update paths based on new EXP_NAME
Config.WORKING_DIR = os.path.join("./working", Config.EXP_NAME)
Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Hyperparameters for speed
Config.SEED = 42
Config.EPOCHS = 1
Config.N_FOLDS = 1  # Run only one fold
Config.TRAIN_BATCH_SIZE = 4
Config.VALID_BATCH_SIZE = 4
Config.GRAD_ACCUM_STEPS = 1
Config.AWP_START_EPOCH = 0  # Enable AWP immediately to test it
Config.NUM_WORKERS = 0  # Disable multiprocessing for simple debug run

# LightGBM fast settings
Config.LGBM_PARAMS["n_estimators"] = 10
Config.LGBM_PARAMS["early_stopping_rounds"] = 5
Config.LGBM_PARAMS["verbosity"] = -1

# Initialize directories and seeds
Config.setup()

# 2. Import Library Modules (AFTER Config updates)
import sys

# Force reload of library modules to ensure code changes are picked up in persistent sessions
for module in ["library.model", "library.trainer", "library.stacking", "library.data"]:
    if module in sys.modules:
        del sys.modules[module]

from library.data import get_dataloaders
from library.model import EssayModel
from library.trainer import train_fold
from library.stacking import train_stacking, predict_stacking
from library.utils import seed_everything


def run_demo():
    seed_everything(Config.SEED)

    # =========================================================================
    # Step 1: Data Pipeline Verification
    # =========================================================================
    print("\n=== Verifying Data Pipeline ===")
    # load_cached_data=False forces the pipeline to process the raw metadata
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # Verify Keys
    required_keys = [
        "input_ids",
        "attention_mask",
        "batch_ids",
        "meta_features",
        "labels",
    ]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Verify Shapes
    # Labels should match batch size
    assert (
        batch["labels"].shape[0] == Config.TRAIN_BATCH_SIZE
    ), f"Label batch size mismatch. Expected {Config.TRAIN_BATCH_SIZE}, got {batch['labels'].shape[0]}"

    # Input IDs should have correct max length
    assert (
        batch["input_ids"].shape[1] == Config.MAX_LENGTH
    ), f"Sequence length mismatch. Expected {Config.MAX_LENGTH}, got {batch['input_ids'].shape[1]}"

    # Meta features should be (Batch, 4)
    assert batch["meta_features"].shape == (
        Config.TRAIN_BATCH_SIZE,
        4,
    ), f"Meta features shape mismatch. Got {batch['meta_features'].shape}"

    print("Data Pipeline check passed.")

    # =========================================================================
    # Step 2: Model Architecture Verification
    # =========================================================================
    print("\n=== Verifying Model Architecture ===")
    model = EssayModel()
    model.to(Config.DEVICE)

    # Prepare inputs
    input_ids = batch["input_ids"].to(Config.DEVICE)
    attention_mask = batch["attention_mask"].to(Config.DEVICE)
    batch_ids = batch["batch_ids"].to(Config.DEVICE)

    # Forward Pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, batch_ids)

    # Verify Outputs
    assert "logits" in outputs and "embeddings" in outputs
    assert outputs["logits"].shape == (
        Config.TRAIN_BATCH_SIZE,
        1,
    ), f"Logits shape mismatch. Got {outputs['logits'].shape}"
    assert outputs["embeddings"].shape == (
        Config.TRAIN_BATCH_SIZE,
        model.config.hidden_size,
    ), f"Embeddings shape mismatch. Got {outputs['embeddings'].shape}"

    print("Model Architecture check passed.")

    # Cleanup memory
    del model, batch, input_ids, attention_mask, batch_ids, outputs
    torch.cuda.empty_cache()

    # =========================================================================
    # Step 3: Training Loop Verification (Fold 0)
    # =========================================================================
    print("\n=== Verifying Training Loop (Fold 0) ===")
    # This runs the full training logic: Forward, Backward, AWP, Validation, Saving
    best_qwk = train_fold(fold_idx=0)

    print(f"Training finished. Best QWK: {best_qwk:.4f}")

    # Verify Artifacts
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "backbone_fold_0.pth")
    oof_embed_path = os.path.join(Config.CACHE_DIR, "oof_embeddings_fold_0.npy")
    oof_target_path = os.path.join(Config.CACHE_DIR, "oof_targets_fold_0.npy")

    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    assert os.path.exists(
        oof_embed_path
    ), f"OOF Embeddings not found at {oof_embed_path}"
    assert os.path.exists(
        oof_target_path
    ), f"OOF Targets not found at {oof_target_path}"

    print("Training Loop check passed.")

    # =========================================================================
    # Step 4: Stacking & Inference Verification
    # =========================================================================
    print("\n=== Verifying Stacking & Inference ===")

    # Train Stacking Model
    # Note: We set N_FOLDS=1, so it will only look for fold 0 OOF data, which we just generated.
    lgbm_model = train_stacking(load_cached_data=True)

    lgbm_path = os.path.join(Config.WORKING_DIR, "lgbm_stacking.txt")
    assert os.path.exists(lgbm_path), "LightGBM model file not saved."

    # Run Inference
    # This uses the trained backbone (fold 0) and the stacking model to predict test set
    predict_stacking(model=lgbm_model, load_cached_data=True)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")

    # Check columns
    assert "essay_id" in sub_df.columns
    assert "score" in sub_df.columns

    # Check values
    scores = sub_df["score"]
    assert (
        scores.min() >= 1 and scores.max() <= 6
    ), "Predictions contain scores outside range [1, 6]"
    assert pd.api.types.is_integer_dtype(scores), "Scores are not integers"

    print("Stacking & Inference check passed.")
    print("\nAll systems operational. Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
