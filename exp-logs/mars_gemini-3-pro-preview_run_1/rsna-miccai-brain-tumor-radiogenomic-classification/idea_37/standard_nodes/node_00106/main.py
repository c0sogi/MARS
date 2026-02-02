import os
import glob
import shutil
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.geometry_utils import process_subject_geometry
from library.engine import train_model
from library.inference import predict_test_set, load_models
from library.dataset import get_dataloader


def main():
    # 1. Setup
    Config.setup()
    Config.NUM_EPOCHS = 5  # Reduce epochs for fast baseline execution

    print("Initializing workflow...")

    # Load metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Pre-compute geometry for full sets
    # This ensures that when we run final inference or use the full sets, the cache is ready.
    print("Pre-computing geometry metadata...")
    process_subject_geometry(train_meta)
    process_subject_geometry(val_meta)
    process_subject_geometry(test_meta)

    # 3. 5-Fold Cross Validation
    print(f"Starting {Config.NUM_FOLDS}-Fold Cross-Validation...")

    # We perform CV on the provided TRAIN set.
    # The provided VAL set will remain as the hold-out for final scoring.

    X = train_meta
    y = train_meta["MGMT_value"]

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Save original paths to restore later
    original_train_path = Config.TRAIN_METADATA_PATH
    original_val_path = Config.VAL_METADATA_PATH

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold} ===")

        # Create temporary fold datasets
        fold_train_df = X.iloc[train_idx].copy()
        fold_val_df = X.iloc[val_idx].copy()

        temp_train_path = os.path.join(Config.WORKING_DIR, f"train_fold_{fold}.csv")
        temp_val_path = os.path.join(Config.WORKING_DIR, f"val_fold_{fold}.csv")

        fold_train_df.to_csv(temp_train_path, index=False)
        fold_val_df.to_csv(temp_val_path, index=False)

        # Clear geometry cache for these specific lengths to prevent subject ID mismatch.
        # The library caches based on len(df), so different folds of the same size would collide.
        for df_len in [len(fold_train_df), len(fold_val_df)]:
            cache_file = os.path.join(
                Config.CACHE_DIR, f"geometry_cache_{df_len}.parquet"
            )
            if os.path.exists(cache_file):
                os.remove(cache_file)

        # Override Config paths to point to fold data
        Config.TRAIN_METADATA_PATH = temp_train_path
        Config.VAL_METADATA_PATH = temp_val_path

        # Train
        # train_model saves to Config.CACHE_DIR/best_model.pth
        best_model_path = train_model(num_epochs=Config.NUM_EPOCHS)

        # Rename to fold specific model
        fold_model_dest = os.path.join(Config.CACHE_DIR, f"best_model_fold{fold}.pth")
        if os.path.exists(best_model_path):
            shutil.move(best_model_path, fold_model_dest)
            print(f"Saved {fold_model_dest}")
        else:
            print("Warning: best_model.pth not found after training.")

    # Restore Config paths
    Config.TRAIN_METADATA_PATH = original_train_path
    Config.VAL_METADATA_PATH = original_val_path

    # 4. Validation Assessment on Hold-out Set
    print("\n=== Hold-out Validation Assessment ===")

    device = torch.device(Config.DEVICE)
    models = load_models(device)

    if not models:
        print("No models found. Exiting.")
        return

    # Load hold-out validation loader
    # We rely on the cached geometry from step 2 (or it recalculates safely since it's the full set)
    val_loader = get_dataloader("val", load_cached_geometry=True)

    all_preds = []
    all_targets = []

    # Inference
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Ensemble prediction
            batch_probs = []
            for model in models:
                logits = model(images)
                probs = torch.sigmoid(logits)
                batch_probs.append(probs.cpu().numpy())

            # Average predictions across models
            avg_probs = np.mean(batch_probs, axis=0)

            all_preds.extend(avg_probs.flatten().tolist())
            all_targets.extend(labels.numpy().flatten().tolist())

    final_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Ensure alignment
    if len(df_val) == len(all_preds):
        df_val["pred"] = all_preds
        df_val["target"] = all_targets
        df_val["error"] = np.abs(df_val["pred"] - df_val["target"])

        # Load geometry features to correlate
        geo_df = process_subject_geometry(df_val, load_cached_data=True)
        analysis_df = pd.merge(df_val, geo_df, on="BraTS21ID")

        # Correlate error with numeric features
        numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
        correlations = (
            analysis_df[numeric_cols].corr()["error"].sort_values(ascending=False)
        )

        print("Top correlations with Error:")
        print(correlations.head(5))
    else:
        print("Mismatch in validation set size during analysis.")

    # 6. Submission
    THRESHOLD = 0.6705454545454544
    if final_auc > THRESHOLD:
        print(f"\nMetric {final_auc} > {THRESHOLD}. Generating submission...")
        predict_test_set(load_cached_geometry=True)
    else:
        print(f"\nMetric {final_auc} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
