import sys
import os
import gc
import numpy as np
import pandas as pd
import warnings
from library.config import Config
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngineer
from library.knowledge_base import KnowledgeBase
from library.model import TaxiFareModel

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Configuration and Setup
    config = Config(debug=False)
    config.set_seed()
    config.setup_dirs()

    print("=== Starting K-Fold Target Encoding Pipeline ===")

    # 2. Data Loading
    loader = DataLoader(config)
    train_df, val_df, test_df = loader.get_data(load_cached_data=True)

    # 3. Feature Engineering
    fe = FeatureEngineer(config)

    # Process Train: Full features
    train_df = fe.process(train_df, cache_key="train", is_background=False)

    # Process Validation & Test
    val_df = fe.process(val_df, cache_key="val", is_background=False)
    test_df = fe.process(test_df, cache_key="test", is_background=False)

    # 4. Knowledge Base (K-Fold Encoding)
    kb = KnowledgeBase(config)

    # Encode Train (K-Fold)
    # This adds 'smart_fare', 'smart_rate' etc. to train_df using out-of-fold stats
    train_df = kb.process_kfold(train_df, load_cached_data=True)

    # Encode Val/Test (Global)
    # This adds 'smart_fare' using global stats from train_df
    val_df = kb.process_test(val_df, train_df)
    test_df = kb.process_test(test_df, train_df)

    # 5. Model Training
    # Subsample training data for XGBoost efficiency
    # We have already encoded the full dataset, so the rows in the subsample
    # contain "Global Knowledge" from the full 44M rows.
    if config.TRAIN_SUBSAMPLE_SIZE and len(train_df) > config.TRAIN_SUBSAMPLE_SIZE:
        print(
            f"Subsampling training set to {config.TRAIN_SUBSAMPLE_SIZE} rows for Model Fitting..."
        )
        train_sub = train_df.sample(
            n=config.TRAIN_SUBSAMPLE_SIZE, random_state=config.SEED
        )
        del train_df
        gc.collect()
    else:
        train_sub = train_df

    model = TaxiFareModel(config)
    model.train(train_sub, val_df)

    # 6. Validation Assessment
    print("\n=== Validation Assessment ===")
    # Generate predictions on the full validation set
    val_preds = model.predict(val_df)
    y_val = val_df["fare_amount"].values

    # Compute RMSE
    rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))
    print(f"Final Validation Metric: {rmse}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(y_val - val_preds)

    # Calculate correlation between error magnitude and numerical features
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns
    correlations = {}

    for col in numeric_cols:
        if col == "fare_amount":
            continue

        # Ensure column has valid data
        col_data = val_df[col]
        valid_mask = ~np.isnan(col_data) & ~np.isnan(errors)

        if valid_mask.sum() > 1:
            try:
                corr = np.corrcoef(col_data[valid_mask], errors[valid_mask])[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr
            except Exception:
                pass

    # Print top correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top Feature Correlations with Error Magnitude:")
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val:.4f}")

    # 8. Submission Generation
    # Threshold defined in task requirements
    TARGET_METRIC = 3.5069767944123895

    if rmse < TARGET_METRIC:
        print(
            f"\nMetric {rmse} meets threshold {TARGET_METRIC}. Generating submission..."
        )
        test_preds = model.predict(test_df)
        model.generate_submission(test_df, test_preds)
    else:
        print(
            f"\nMetric {rmse} does NOT meet threshold {TARGET_METRIC}. Submission skipped."
        )


if __name__ == "__main__":
    main()
