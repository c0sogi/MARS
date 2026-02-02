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
    # Using debug=False to ensure we use the full 5M foreground / 38M background split
    # needed to achieve the high precision metric.
    # The A100 GPU is sufficient to train this in minutes.
    config = Config(debug=False)
    config.set_seed()
    config.setup_dirs()

    print("=== Starting Disjoint Background-Foreground Pipeline ===")

    # 2. Data Loading
    # Loads raw data and performs the disjoint split
    loader = DataLoader(config)
    bg_df, fg_df, val_df, test_df = loader.get_data(load_cached_data=True)

    # 3. Feature Engineering
    fe = FeatureEngineer(config)

    # Process Background: Minimal features (keys) for aggregation
    bg_df = fe.process(bg_df, cache_key="background", is_background=True)

    # Process Foreground (Train): Full features
    fg_df = fe.process(fg_df, cache_key="foreground", is_background=False)

    # Process Validation: Full features
    val_df = fe.process(val_df, cache_key="val", is_background=False)

    # Process Test: Full features
    test_df = fe.process(test_df, cache_key="test", is_background=False)

    # 4. Knowledge Base Construction & Enrichment
    kb = KnowledgeBase(config)

    # Compute Priors from the massive Background set
    priors = kb.compute_priors(bg_df, load_cached_data=True)

    # Release Background memory immediately after computing priors
    del bg_df
    gc.collect()

    # Enrich datasets with the computed priors (Left Join + Smart Fallback)
    fg_df = kb.enrich_dataset(fg_df, priors)
    val_df = kb.enrich_dataset(val_df, priors)
    test_df = kb.enrich_dataset(test_df, priors)

    # 5. Model Training
    model = TaxiFareModel(config)
    model.train(fg_df, val_df)

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
