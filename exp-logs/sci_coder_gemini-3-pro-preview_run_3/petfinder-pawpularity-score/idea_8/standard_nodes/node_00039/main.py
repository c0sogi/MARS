import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.dataset import PetDataset
from library.extractors import FeatureExtractor
from library.preprocessor import FeaturePreprocessor
from library.ensemble import StackingEnsemble
from library.utils import seed_everything, get_device


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading metadata...")
    if not os.path.exists(Config.TRAIN_META_PATH):
        raise FileNotFoundError(f"Train metadata not found at {Config.TRAIN_META_PATH}")

    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # Create Datasets
    # We use tta=True for feature extraction as per Idea description
    train_dataset = PetDataset(df_train, mode="train", tta=True)
    val_dataset = PetDataset(df_val, mode="val", tta=True)
    test_dataset = PetDataset(df_test, mode="test", tta=True)

    # Create DataLoaders
    # num_workers=4, pin_memory=True for speed
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Feature Extraction
    print("Starting Feature Extraction...")
    extractor = FeatureExtractor()
    # load_cached_data=True is critical to use existing features if available
    raw_features = extractor.extract_all(
        train_loader, val_loader, test_loader, load_cached_data=True
    )

    # 4. Feature Preprocessing
    print("Starting Feature Preprocessing...")
    preprocessor = FeaturePreprocessor()
    X_train, y_train, X_val, y_val, X_test, test_ids = preprocessor.preprocess(
        raw_features, load_cached_data=True
    )

    # 5. Validation Phase
    print("Starting Validation Phase...")
    ensemble = StackingEnsemble()

    # Train on Train set, Validate on Val set (Hold-out)
    # Step A: Generate OOF preds for Train set to train Meta-Learner
    print("Generating OOF predictions on Train set...")
    oof_train = ensemble.cross_validate(X_train, y_train)

    # Step B: Train models on Train set
    print("Training ensemble on Train set...")
    ensemble.fit_final(X_train, y_train, oof_train)

    # Step C: Predict on Hold-out Val set
    print("Predicting on Hold-out Validation set...")
    # Access internal models from ensemble
    p_svr = ensemble.final_svr.predict(X_val)
    p_et = ensemble.final_et.predict(X_val)
    p_lgbm = ensemble.final_lgbm.predict(X_val)

    # Stack base predictions
    base_val_preds = np.column_stack([p_svr, p_et, p_lgbm])

    # Meta-learner prediction
    val_preds = ensemble.meta_learner.predict(base_val_preds)
    val_preds = np.clip(val_preds, 1.0, 100.0)

    # Calculate Metric
    final_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Final Validation Metric: {final_rmse}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    residuals = np.abs(y_val - val_preds)

    # Correlate residuals with metadata features
    # We use df_val loaded earlier
    meta_cols = Config.METADATA_COLS
    correlations = {}
    for col in meta_cols:
        if col in df_val.columns:
            feat_vals = df_val[col].values
            # Point-biserial correlation is essentially Pearson for binary-continuous
            corr, _ = pearsonr(feat_vals, residuals)
            correlations[col] = corr

    # Sort and print
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Correlation between Absolute Error and Metadata Features:")
    for name, val in sorted_corrs:
        print(f"  {name}: {val:.4f}")

    # 7. Submission Logic
    THRESHOLD = 17.361083072547856

    if final_rmse < THRESHOLD:
        print(
            f"\nValidation Metric ({final_rmse}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Combine Train and Val
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        # Generate OOF for full set (needed for meta learner training)
        print("Generating OOF predictions on Full (Train+Val) set...")
        oof_full = ensemble.cross_validate(X_full, y_full)

        # Train Final Models
        print("Retraining ensemble on Full set...")
        ensemble.fit_final(X_full, y_full, oof_full)

        # Predict on Test
        ensemble.predict_and_submit(X_test, test_ids)

    else:
        print(
            f"\nValidation Metric ({final_rmse}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
