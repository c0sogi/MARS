import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, TargetScaler
from library.data_processing import process_and_cache_features
from library.dataset import VolcanoDataset
from library.model import ChannelAdaptiveHybridModel
from library.trainer import Trainer

# Suppress warnings
warnings.filterwarnings("ignore")


def run_pipeline():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for Fast Baseline
    Config.EPOCHS = 15
    Config.setup()
    seed_everything(Config.SEED)

    print(f"Running on device: {Config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    # Load Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA)
    df_val_meta = pd.read_csv(Config.VAL_METADATA)

    # Process Statistical Features (Train/Val)
    # This reads CSVs, computes stats, and caches them to Parquet
    print("Processing training features...")
    df_train_stats = process_and_cache_features(
        Config.TRAIN_METADATA, Config.TRAIN_FEATURES_PATH, load_cached_data=True
    )

    print("Processing validation features...")
    df_val_stats = process_and_cache_features(
        Config.VAL_METADATA, Config.VAL_FEATURES_PATH, load_cached_data=True
    )

    # ---------------------------------------------------------
    # 3. Scalers
    # ---------------------------------------------------------
    print("Fitting scalers...")
    # Target Scaler (Time to Eruption)
    target_scaler = TargetScaler()
    target_scaler.fit(df_train_meta["time_to_eruption"].values)

    # Stats Scaler (Input Features)
    # Identify feature columns (exclude ID and target)
    feature_cols = sorted(
        [
            c
            for c in df_train_stats.columns
            if c not in ["segment_id", "time_to_eruption"]
        ]
    )

    stats_scaler = StandardScaler()
    stats_scaler.fit(df_train_stats[feature_cols].values)

    # ---------------------------------------------------------
    # 4. Datasets & Loaders
    # ---------------------------------------------------------
    train_dataset = VolcanoDataset(
        metadata_df=df_train_meta,
        stats_df=df_train_stats,
        augment=True,
        target_scaler=target_scaler,
        stats_scaler=stats_scaler,
    )

    val_dataset = VolcanoDataset(
        metadata_df=df_val_meta,
        stats_df=df_val_stats,
        augment=False,
        target_scaler=target_scaler,
        stats_scaler=stats_scaler,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 5. Training
    # ---------------------------------------------------------
    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        target_scaler=target_scaler,
        num_stats_features=len(feature_cols),
    )

    trainer.fit()

    # ---------------------------------------------------------
    # 6. Evaluation & Metric
    # ---------------------------------------------------------
    print("Loading best model for evaluation...")
    model = ChannelAdaptiveHybridModel(num_stats_features=len(feature_cols))
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()

    val_preds = []
    val_targets = []

    # Inference on Validation Set
    with torch.no_grad():
        for spectrogram, stats, target in val_loader:
            spectrogram = spectrogram.to(Config.DEVICE)
            stats = stats.to(Config.DEVICE)

            output = model(spectrogram, stats)

            # Move to CPU numpy
            pred_batch = output.cpu().numpy().flatten()
            target_batch = (
                target.numpy().flatten()
            )  # Target from loader is already scaled

            # Inverse Scale
            pred_orig = target_scaler.inverse_transform(pred_batch)
            target_orig = target_scaler.inverse_transform(target_batch)

            val_preds.extend(pred_orig)
            val_targets.extend(target_orig)

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate Metric
    mae = np.mean(np.abs(val_preds - val_targets))
    print(f"Final Validation Metric: {mae}")

    # ---------------------------------------------------------
    # 7. Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Failure Analysis ---")
    errors = np.abs(val_preds - val_targets)

    # Align errors with features
    # df_val_meta order matches val_loader (shuffle=False)
    df_analysis = df_val_meta.copy()
    df_analysis["abs_error"] = errors

    # Merge with stats features
    df_analysis = df_analysis.merge(df_val_stats, on="segment_id", how="left")

    # Calculate correlations
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns
    correlations = (
        df_analysis[numeric_cols].corr()["abs_error"].abs().sort_values(ascending=False)
    )

    print("Top 5 Features Correlated with Error Magnitude:")
    # Skip the first one if it is 'abs_error' itself
    print(correlations.head(6))

    # ---------------------------------------------------------
    # 8. Submission
    # ---------------------------------------------------------
    THRESHOLD = 1492505.6322055138

    if mae < THRESHOLD:
        print(f"\nMetric {mae} < Threshold {THRESHOLD}. Generating submission...")

        # Load Test Data
        df_test_meta = pd.read_csv(Config.TEST_METADATA)

        print("Processing test features...")
        df_test_stats = process_and_cache_features(
            Config.TEST_METADATA, Config.TEST_FEATURES_PATH, load_cached_data=True
        )

        # Test Dataset
        test_dataset = VolcanoDataset(
            metadata_df=df_test_meta,
            stats_df=df_test_stats,
            augment=False,
            target_scaler=None,  # No target in test
            stats_scaler=stats_scaler,  # Use same scaler as train
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds = []

        # Inference
        with torch.no_grad():
            for spectrogram, stats in test_loader:
                spectrogram = spectrogram.to(Config.DEVICE)
                stats = stats.to(Config.DEVICE)

                output = model(spectrogram, stats)
                pred_batch = output.cpu().numpy().flatten()

                # Inverse Scale
                pred_orig = target_scaler.inverse_transform(pred_batch)
                test_preds.extend(pred_orig)

        # Save Submission
        df_sub = pd.DataFrame(
            {"segment_id": df_test_meta["segment_id"], "time_to_eruption": test_preds}
        )

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {mae} >= Threshold {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    run_pipeline()
