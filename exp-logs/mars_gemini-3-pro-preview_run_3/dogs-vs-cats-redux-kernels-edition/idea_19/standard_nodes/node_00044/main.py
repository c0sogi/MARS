import os
import pandas as pd
import numpy as np
import torch
import glob
from scipy.stats import pointbiserialr

# Import from library
from library.config import (
    MODEL_CONFIGS,
    WORKING_DIR,
    SUBMISSION_PATH,
    DEVICE,
    SEED,
    TRAIN_META_PATH,
    VAL_META_PATH,
    OOF_THRESHOLD,
    ModelConfig,
)
from library.data import get_data_splits, get_fold_loaders, get_test_loader
from library.engine import run_fold
from library.utils import (
    seed_everything,
    compute_log_loss,
    load_checkpoint,
    save_submission,
)
from library.models import create_model
from library.inference import predict


def analyze_failures(df, pred_col, target_col, metadata_dir="./metadata"):
    """
    Performs failure analysis by correlating error with metadata features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate Error
    df["error"] = np.abs(df[target_col] - df[pred_col])

    # We need to merge with original metadata to get width/height/file_size
    # The df here comes from get_data_splits which has 'filepath'
    # We can re-read the train/val csvs to get filenames if needed,
    # but let's extract metadata on the fly or assume we can get it.
    # Since reading all images is slow, we will check if we can get file size easily.

    # Construct full paths
    input_dir = "./input"

    # We'll calculate file size and aspect ratio for the validation set
    # This might be slow for 25k images, but we only do it for the validation set (which is full train here effectively)
    # Actually, failure analysis is usually on the validation set.

    print("Computing metadata features for correlation analysis...")
    file_sizes = []

    # To save time, we'll just use file_size as a proxy for complexity/quality
    # and maybe parse resolution if easily available.
    # For this baseline, file_size is quick.

    for idx, row in df.iterrows():
        full_path = os.path.join(input_dir, row["filepath"])
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path))
        else:
            file_sizes.append(0)

    df["file_size"] = file_sizes

    # Correlation
    corr, pval = pointbiserialr(df["error"], df["file_size"])
    print(f"Correlation between Error and File Size: {corr:.4f} (p-value: {pval:.4f})")

    # If we had width/height in the split dataframe, we would use them.
    # Since they aren't in the standard metadata csvs provided in the description (only filepath, label, filename),
    # we stick to file_size which is a strong proxy for resolution/quality.


def main():
    # 1. Setup
    seed_everything(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Override Configs for Fast Baseline
    # Reducing epochs to 2 to ensure completion within 2 hours for 15 models
    print("Configuring models for fast baseline execution (Epochs=2)...")
    for cfg in MODEL_CONFIGS:
        cfg.epochs = 2

    # 2. Prepare Data & Storage
    # Load full dataset with fold assignments
    df_folds = get_data_splits(load_cached_data=True)

    # Dictionary to store OOF predictions for each model architecture
    # Key: model_name, Value: numpy array of shape (len(df_folds),) filled with nans initially
    model_oof_preds = {cfg.model_name: np.zeros(len(df_folds)) for cfg in MODEL_CONFIGS}

    # Keep track of which indices were filled to ensure integrity
    filled_indices = {
        cfg.model_name: np.zeros(len(df_folds), dtype=bool) for cfg in MODEL_CONFIGS
    }

    # 3. Training & OOF Inference Loop
    print("\n=== Starting Training & OOF Inference Loop ===")

    for cfg in MODEL_CONFIGS:
        print(f"\n>> Processing Architecture: {cfg.model_name}")

        for fold_idx in range(5):  # 5 Folds
            # A. Train
            # run_fold returns best_val_loss, but we rely on the saved checkpoint
            _ = run_fold(fold_idx, cfg)

            # B. OOF Inference
            print(f"   Generating OOF predictions for Fold {fold_idx}...")

            # Load Validation Loader for this fold
            # We reuse get_fold_loaders which returns (train, val). We only need val.
            _, val_loader = get_fold_loaders(fold_idx, cfg, load_cached_data=True)

            # Load Best Checkpoint
            model = create_model(
                cfg, pretrained=False
            )  # Weights loaded from checkpoint
            model = model.to(DEVICE)
            checkpoint_name = f"{cfg.model_name}_fold_{fold_idx}.pth"
            load_checkpoint(checkpoint_name, model, device=DEVICE)

            # Predict
            probs, targets = predict(model, val_loader, device=DEVICE)

            # Align predictions with the dataframe
            # val_loader comes from df[df['fold'] == fold_idx]
            # We need to assign these probs to the corresponding indices in df_folds
            val_indices = df_folds[df_folds["fold"] == fold_idx].index

            if len(probs) != len(val_indices):
                raise ValueError(
                    f"Mismatch in prediction length: {len(probs)} vs {len(val_indices)}"
                )

            model_oof_preds[cfg.model_name][val_indices] = probs
            filled_indices[cfg.model_name][val_indices] = True

            # Cleanup to save memory
            del model
            torch.cuda.empty_cache()

    # 4. Quality Gating & Ensemble
    print("\n=== Quality Gating & Ensemble Aggregation ===")

    valid_models = []
    final_ensemble_preds = np.zeros(len(df_folds))

    for cfg in MODEL_CONFIGS:
        name = cfg.model_name
        preds = model_oof_preds[name]

        # Check if we have predictions for all samples
        if not np.all(filled_indices[name]):
            print(f"Warning: Model {name} has missing OOF predictions. Skipping.")
            continue

        # Calculate Log Loss
        loss = compute_log_loss(df_folds["label"].values, preds)
        print(f"Model: {name} | OOF Log Loss: {loss:.6f}")

        if loss < OOF_THRESHOLD:
            valid_models.append(name)
            final_ensemble_preds += preds
        else:
            print(f"  -> Excluded (Threshold > {OOF_THRESHOLD})")

    if not valid_models:
        print("Error: No models passed the quality gate. Using all models as fallback.")
        # Fallback: average all
        final_ensemble_preds = np.zeros(len(df_folds))
        for cfg in MODEL_CONFIGS:
            final_ensemble_preds += model_oof_preds[cfg.model_name]
        valid_models = [cfg.model_name for cfg in MODEL_CONFIGS]

    # Compute Average
    final_ensemble_preds /= len(valid_models)

    # 5. Final Metric & Analysis
    final_metric = compute_log_loss(df_folds["label"].values, final_ensemble_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Add predictions to dataframe for analysis
    df_folds["ensemble_pred"] = final_ensemble_preds
    analyze_failures(df_folds, "ensemble_pred", "label")

    # 6. Submission Logic
    # Threshold from task description
    SUBMISSION_THRESHOLD = 0.009074434935821756

    if final_metric < SUBMISSION_THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) meets threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        test_preds_accumulator = None
        test_ids = None

        # We need to iterate through all valid models and all their folds
        total_inference_runs = 0

        for cfg in MODEL_CONFIGS:
            if cfg.model_name not in valid_models:
                continue

            # Get Test Loader (Transform depends on config)
            test_loader = get_test_loader(cfg)

            for fold_idx in range(5):
                print(f"Inference: {cfg.model_name} (Fold {fold_idx})")

                # Load Model
                model = create_model(cfg, pretrained=False)
                model = model.to(DEVICE)
                checkpoint_name = f"{cfg.model_name}_fold_{fold_idx}.pth"

                try:
                    load_checkpoint(checkpoint_name, model, device=DEVICE)
                except FileNotFoundError:
                    print(f"Warning: Checkpoint {checkpoint_name} not found. Skipping.")
                    continue

                # Predict
                probs, ids = predict(model, test_loader, device=DEVICE)

                # Initialize accumulator if first run
                if test_preds_accumulator is None:
                    test_preds_accumulator = np.zeros_like(probs)
                    test_ids = ids

                test_preds_accumulator += probs
                total_inference_runs += 1

                del model
                torch.cuda.empty_cache()

        if total_inference_runs > 0:
            avg_test_preds = test_preds_accumulator / total_inference_runs
            save_submission(test_ids, avg_test_preds, SUBMISSION_PATH)
            print(f"Submission saved to {SUBMISSION_PATH}")
        else:
            print("Error: No inference runs completed.")

    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not meet threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
