import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil
from torch.utils.data import DataLoader

# Add current directory to path to ensure library imports work
sys.path.append(".")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import WhaleDataset, WhaleTransforms
from library.models import WhaleClassifier
from library.trainer import run_fold
from library.stacking import (
    train_meta_learner,
    predict_meta_learner,
    create_submission_file,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed & Demonstration
    # -------------------------------------------------------------------------
    print("\n[1] Configuring Environment...")

    # Force Debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 64  # Small enough for speed, large enough for batches

    # Reduce training intensity
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_FOLDS = 2  # We will only run Fold 0 for this demo
    Config.MODELS = ["tf_efficientnet_b0_ns"]  # Use the smallest model

    # Ensure reproducible results
    seed_everything(Config.SEED)

    # Clean working directory for a fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Dataset Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Dataset Loading...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLES].copy()
        test_df = test_df.iloc[: Config.DEBUG_SAMPLES].copy()
        print(
            f"Debug Mode: Loaded {len(train_df)} train samples and {len(test_df)} test samples."
        )

    # Instantiate Dataset
    ds = WhaleDataset(
        train_df, split_name="demo_train", transform=WhaleTransforms(mode="train")
    )

    # Check __getitem__
    spec, label = ds[0]
    print(f"Spectrogram Shape: {spec.shape}")
    print(f"Label: {label}")

    # Assertions
    assert spec.ndim == 3, "Spectrogram must be 3D (Channels, Freq, Time)"
    assert spec.shape[0] == 1, "Input channels should be 1"
    assert isinstance(label, torch.Tensor), "Label must be a tensor"

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model_name = Config.MODELS[0]
    model = WhaleClassifier(model_name, pretrained=False)  # False for speed in demo
    model.to(Config.DEVICE)
    model.eval()

    # Dummy forward pass
    dummy_input = torch.randn(2, 1, 128, 63).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), "Output shape mismatch (Batch, Num_Classes)"

    del model, dummy_input
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration (Fold 0)
    # -------------------------------------------------------------------------
    print("\n[4] Running Training for Fold 0...")

    # This function handles the training loop and saves checkpoints
    run_fold(fold_idx=0, model_name=model_name, df=train_df)

    # Verify checkpoint creation
    ckpt_path = os.path.join(Config.WORKING_DIR, f"{model_name}_fold_0_best_auc.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    print(f"Checkpoint successfully saved: {ckpt_path}")

    # -------------------------------------------------------------------------
    # 5. Inference & Stacking Preparation
    # -------------------------------------------------------------------------
    print("\n[5] Generating Predictions for Stacking...")

    # We need OOF predictions (Validation) and Test predictions
    # Re-load the best model from Fold 0
    model = WhaleClassifier(model_name, pretrained=False)
    checkpoint = torch.load(ckpt_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(Config.DEVICE)
    model.eval()

    # --- Generate OOF Preds (on Validation subset of Fold 0) ---
    # We need to recreate the validation split used in run_fold
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    _, val_idx = list(skf.split(train_df, train_df["label"]))[0]
    val_df = train_df.iloc[val_idx].copy()

    val_dataset = WhaleDataset(val_df, split_name="demo_val_fold_0", transform=None)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    oof_preds = []
    oof_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(Config.DEVICE)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            oof_preds.extend(probs)
            oof_targets.extend(targets.numpy())

    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)

    print(f"Generated {len(oof_preds)} OOF predictions.")
    print(f"OOF AUC: {get_score(oof_targets, oof_preds):.4f}")

    # --- Generate Test Preds ---
    test_dataset = WhaleDataset(test_df, split_name="demo_test", transform=None)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    test_preds_arr = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(Config.DEVICE)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            test_preds_arr.extend(probs)

    test_preds_arr = np.array(test_preds_arr)
    print(f"Generated {len(test_preds_arr)} Test predictions.")

    # Clean up
    del model, val_loader, test_loader
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 6. Meta-Learner (Stacking)
    # -------------------------------------------------------------------------
    print("\n[6] Running Meta-Learner...")

    # Prepare dictionaries as expected by stacking.py
    # In a real scenario, we would have preds from multiple models/folds
    oof_dict = {model_name: oof_preds}
    test_dict = {model_name: test_preds_arr}

    # Train Meta-Learner
    # Note: We set load_cached_data=False to force re-computation for this demo
    meta_weights = train_meta_learner(oof_dict, oof_targets, load_cached_data=False)

    assert "coef" in meta_weights, "Meta-learner did not return coefficients"
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "meta_learner_coef.npy")
    ), "Meta-learner weights not saved"

    # Predict with Meta-Learner
    final_probs = predict_meta_learner(test_dict, load_cached_data=False)

    assert len(final_probs) == len(test_df), "Final predictions count mismatch"
    print(f"Meta-learner prediction mean: {final_probs.mean():.4f}")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[7] Creating Submission File...")

    clip_names = test_df["clip"].values
    create_submission_file(clip_names, final_probs)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Validate submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(sub_df.columns) == [
        "clip",
        "probability",
    ], "Submission columns mismatch"
    assert len(sub_df) == len(test_df), "Submission row count mismatch"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
