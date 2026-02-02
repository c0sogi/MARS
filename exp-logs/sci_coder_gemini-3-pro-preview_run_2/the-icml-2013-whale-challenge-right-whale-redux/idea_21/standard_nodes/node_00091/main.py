import os
import glob
import shutil
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import library modules
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

# =============================================================================
# Configuration Overrides for Fast Baseline
# =============================================================================
# Limit execution for the 1-hour time constraint
Config.DEBUG = True
Config.DEBUG_SAMPLES = 200  # Small subset for speed
Config.EPOCHS = 3  # Few epochs for speed
Config.NUM_FOLDS = 2  # Run only 2 folds for demonstration
Config.BATCH_SIZE = 16  # Safe batch size

# =============================================================================
# Helper Functions
# =============================================================================


def inference(model_name, checkpoint_path, df, split_name):
    """
    Loads a model checkpoint and runs inference on a dataset.
    Returns probabilities.
    """
    device = torch.device(Config.DEVICE)

    # Load Model
    model = WhaleClassifier(model_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Prepare Dataset & Loader
    dataset = WhaleDataset(
        df, split_name=split_name, transform=None, load_cached_data=True
    )
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    preds = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds.extend(probs.cpu().numpy())

    return np.array(preds).flatten()


def analyze_failures(val_df, y_true, y_pred):
    """
    Performs failure analysis by correlating error with signal features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate Error Magnitude
    errors = np.abs(y_true - y_pred)

    # Extract Signal Features for Validation Set
    durations = []
    rms_values = []
    peaks = []

    print("Extracting signal features for failure analysis...")
    for idx, row in val_df.iterrows():
        path = os.path.join(Config.INPUT_ROOT, row["file_path"])
        try:
            data, sr = sf.read(path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            durations.append(len(data) / sr)
            rms_values.append(np.sqrt(np.mean(data**2)))
            peaks.append(np.max(np.abs(data)))
        except:
            durations.append(0)
            rms_values.append(0)
            peaks.append(0)

    # Calculate Correlations
    df_analysis = pd.DataFrame(
        {"error": errors, "duration": durations, "rms": rms_values, "peak": peaks}
    )

    print("Correlation between Error and Features:")
    for col in ["duration", "rms", "peak"]:
        corr = df_analysis["error"].corr(df_analysis[col])
        print(f"Error vs {col.capitalize()}: {corr:.4f}")


# =============================================================================
# Main Orchestration
# =============================================================================


def main():
    seed_everything(Config.SEED)

    # 1. Load Metadata
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        print(f"DEBUG Mode: Truncating datasets to {Config.DEBUG_SAMPLES} samples.")
        train_df = train_df.iloc[: Config.DEBUG_SAMPLES].copy()
        val_df = val_df.iloc[: Config.DEBUG_SAMPLES].copy()
        test_df = test_df.iloc[: Config.DEBUG_SAMPLES].copy()

    # Storage for predictions
    # Structure: { 'model_name_obj': pred_array }
    round1_oof_preds = {}
    round1_test_preds = {}

    # =========================================================================
    # ROUND 1: Supervised Training
    # =========================================================================
    print("\n" + "=" * 40)
    print(" ROUND 1: Supervised Training")
    print("=" * 40)

    for fold in range(Config.NUM_FOLDS):
        for model_name in Config.MODELS:
            # Train
            run_fold(fold, model_name, train_df)

            # Checkpoints to process
            checkpoints = ["best_auc", "best_loss"]

            for ckpt_type in checkpoints:
                ckpt_filename = f"{model_name}_fold_{fold}_{ckpt_type}.pth"
                ckpt_path = os.path.join(Config.WORKING_DIR, ckpt_filename)

                if not os.path.exists(ckpt_path):
                    continue

                # Unique key for stacking
                key = f"{model_name}_{ckpt_type}_fold{fold}"

                # Inference on Validation (OOF)
                # Note: In a real OOF scenario, we predict on the fold's val set.
                # Here, for simplicity and stacking consistency, we predict on the fixed hold-out val set.
                print(f"Inference (Round 1) | {key} | Val & Test")
                val_probs = inference(model_name, ckpt_path, val_df, f"val_r1_{fold}")
                test_probs = inference(
                    model_name, ckpt_path, test_df, f"test_r1_{fold}"
                )

                round1_oof_preds[key] = val_probs
                round1_test_preds[key] = test_probs

                # Rename checkpoint to preserve for history
                new_name = f"{model_name}_fold_{fold}_{ckpt_type}_round1.pth"
                shutil.move(ckpt_path, os.path.join(Config.WORKING_DIR, new_name))

    # =========================================================================
    # ROUND 1: Meta-Learning & Pseudo-Labeling
    # =========================================================================
    print("\n[Round 1] Training Meta-Learner...")
    y_val = val_df["label"].values
    _ = train_meta_learner(round1_oof_preds, y_val, load_cached_data=False)

    print("[Round 1] Predicting Test Set...")
    r1_test_probs = predict_meta_learner(round1_test_preds, load_cached_data=False)

    # Pseudo-Labeling
    print("\nGenerating Pseudo-Labels...")
    high_conf_mask = (r1_test_probs > 0.95) | (r1_test_probs < 0.05)
    pseudo_df = test_df[high_conf_mask].copy()

    # Assign labels
    pseudo_labels = (r1_test_probs[high_conf_mask] > 0.5).astype(int)
    pseudo_df["label"] = pseudo_labels

    # Keep only necessary columns
    pseudo_df = pseudo_df[["file_path", "label", "clip"]]

    print(f"Selected {len(pseudo_df)} pseudo-labeled samples.")

    # Combine Datasets
    combined_train_df = pd.concat([train_df, pseudo_df], axis=0).reset_index(drop=True)
    print(f"New Training Set Size: {len(combined_train_df)}")

    # =========================================================================
    # ROUND 2: Self-Distillation (Retraining)
    # =========================================================================
    print("\n" + "=" * 40)
    print(" ROUND 2: Self-Distillation")
    print("=" * 40)

    round2_oof_preds = {}
    round2_test_preds = {}

    for fold in range(Config.NUM_FOLDS):
        for model_name in Config.MODELS:
            # Train on Combined Data
            run_fold(fold, model_name, combined_train_df)

            checkpoints = ["best_auc", "best_loss"]

            for ckpt_type in checkpoints:
                ckpt_filename = f"{model_name}_fold_{fold}_{ckpt_type}.pth"
                ckpt_path = os.path.join(Config.WORKING_DIR, ckpt_filename)

                if not os.path.exists(ckpt_path):
                    continue

                key = f"{model_name}_{ckpt_type}_fold{fold}"

                # Inference on Original Validation Set (Critical for valid metric)
                print(f"Inference (Round 2) | {key} | Val & Test")
                val_probs = inference(model_name, ckpt_path, val_df, f"val_r2_{fold}")
                test_probs = inference(
                    model_name, ckpt_path, test_df, f"test_r2_{fold}"
                )

                round2_oof_preds[key] = val_probs
                round2_test_preds[key] = test_probs

    # =========================================================================
    # Final Stacking & Validation
    # =========================================================================
    print("\n[Round 2] Training Final Meta-Learner...")
    # Train on Round 2 OOFs (predictions on original val set)
    # Note: We use the val set as the "training" set for the stacker here because
    # we don't have a third hold-out set. In a full competition, we'd use nested CV.
    meta_results = train_meta_learner(round2_oof_preds, y_val, load_cached_data=False)

    # Get Final Predictions on Val Set (to compute metric)
    # Cite debug_lesson_16: Ensure Consumption of Refactored API Returns
    # Cite debug_lesson_12: Strictly Segregate Hold-Out Data When Training Stacked Meta-Learners
    # We use the 'oof_preds' returned by the meta-learner training function.
    # These are generated via internal Cross-Validation, ensuring no data leakage.
    final_val_probs = meta_results["oof_preds"]

    # Compute Metric
    final_auc = get_score(y_val, final_val_probs)
    print(f"Final Validation Metric: {final_auc:.16f}")

    # Failure Analysis
    analyze_failures(val_df, y_val, final_val_probs)

    # =========================================================================
    # Submission
    # =========================================================================
    THRESHOLD = 0.9998881660199745

    if final_auc > THRESHOLD:
        print("\nMetric threshold passed. Generating submission...")
        final_test_probs = predict_meta_learner(
            round2_test_preds, load_cached_data=False
        )
        create_submission_file(test_df["clip"].values, final_test_probs)
    else:
        print(
            f"\nMetric {final_auc:.6f} did not pass threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
