import os
import gc
import numpy as np
import pandas as pd
from library import config, data_loader, feature_engineering, model, utils


def main():
    # Set seed for reproducibility
    np.random.seed(config.RANDOM_STATE)

    # -------------------------------------------------------------------------
    # 1. Data Loading & Feature Engineering
    # -------------------------------------------------------------------------
    print("Step 1: Loading and processing data...")

    # Load Metadata
    train_meta = data_loader.load_metadata("train")
    val_meta = data_loader.load_metadata("val")

    # Generate Features (VH-FASE Pipeline)
    # This uses caching, so it's efficient if run multiple times.
    # It generates Level 0 (Distance), Level 1 (Local), Level 2 (Extended), and Field Projections.
    train_df = feature_engineering.generate_hierarchical_features(train_meta, "train")
    val_df = feature_engineering.generate_hierarchical_features(val_meta, "val")

    # Clean up metadata to save memory
    del train_meta, val_meta
    gc.collect()

    # -------------------------------------------------------------------------
    # 2. Subsampling for Fast Baseline
    # -------------------------------------------------------------------------
    # Limit training data to ensure completion within the time limit.
    # We use stratified sampling to preserve the distribution of coupling types.
    MAX_TRAIN_SAMPLES = 500000

    if len(train_df) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data to {MAX_TRAIN_SAMPLES} samples (Stratified)..."
        )
        frac = MAX_TRAIN_SAMPLES / len(train_df)
        # Use groupby sample for stratification
        train_df_sampled = train_df.groupby("type").sample(
            frac=frac, random_state=config.RANDOM_STATE
        )
        print(f"Subsampled shape: {train_df_sampled.shape}")
    else:
        train_df_sampled = train_df

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("\nStep 2: Training Stratified Ensemble...")

    # Adjust Hyperparameters for Speed
    # Reduce n_estimators and ensure GPU usage to meet the "fast baseline" requirement
    for type_name in config.TYPE_SPECIFIC_PARAMS:
        config.TYPE_SPECIFIC_PARAMS[type_name][
            "n_estimators"
        ] = 2500  # Cap at 2500 trees
        config.TYPE_SPECIFIC_PARAMS[type_name]["device"] = "cuda"
        config.TYPE_SPECIFIC_PARAMS[type_name]["tree_method"] = "hist"

    ensemble = model.StratifiedEnsemble()
    ensemble.fit(train_df_sampled, val_df)

    # -------------------------------------------------------------------------
    # 4. Validation & Metrics
    # -------------------------------------------------------------------------
    print("\nStep 3: Validation Inference...")
    val_preds = ensemble.predict(val_df)

    final_metric, type_metrics = utils.calculate_log_mae(val_df, val_preds)

    print(f"Final Validation Metric: {final_metric}")
    print("Metrics by type:", type_metrics)

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nStep 4: Failure Analysis...")
    val_df["prediction"] = val_preds
    val_df["abs_error"] = (
        val_df["scalar_coupling_constant"] - val_df["prediction"]
    ).abs()

    # Correlation Analysis
    # Identify which features correlate most with the model's errors
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns
    exclude_cols = ["id", "scalar_coupling_constant", "prediction", "abs_error"]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = (
        val_df[feature_cols]
        .corrwith(val_df["abs_error"])
        .abs()
        .sort_values(ascending=False)
    )

    print("Top 10 Features correlated with Absolute Error:")
    print(correlations.head(10))

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = -0.7386035268505905

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_meta = data_loader.load_metadata("test")

        # Generate Features for Test
        test_df = feature_engineering.generate_hierarchical_features(test_meta, "test")

        # Predict
        test_preds = ensemble.predict(test_df)

        # Save Submission
        utils.format_submission(test_df["id"], test_preds)
    else:
        print(
            f"\nMetric ({final_metric}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
