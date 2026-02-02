import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_processor as data_processor
import library.trainer as trainer


def main():
    # 1. Setup and Configuration
    print("Initializing pipeline...")
    utils.seed_everything(config.SEED)

    # Runtime configuration overrides
    # Using lower learning rate and higher estimators for better convergence with granular features
    # Cite solution_lesson_node_00064
    config.LGBM_PARAMS["n_estimators"] = 10000
    config.LGBM_PARAMS["learning_rate"] = 0.01
    config.LGBM_PARAMS["verbosity"] = -1

    # 2. Data Processing
    print("Loading/Processing datasets...")

    # Load Training Data
    train_meta_path = os.path.join(config.METADATA_DIR, "train.csv")
    train_df = data_processor.process_set(
        metadata_path=train_meta_path,
        cache_name="train_features.parquet",
        load_cached_data=True,
    )

    # Load Hold-out Validation Data
    val_meta_path = os.path.join(config.METADATA_DIR, "val.csv")
    val_df = data_processor.process_set(
        metadata_path=val_meta_path,
        cache_name="val_features.parquet",
        load_cached_data=True,
    )

    # 3. Model Training
    # We train on the training set and will evaluate on the hold-out validation set
    print("Training model ensemble...")
    models = trainer.run_cross_validation(train_df)

    # 4. Validation Inference
    print("Performing validation inference...")
    X_val = val_df.drop(columns=["segment_id", "time_to_eruption"])
    y_val = val_df["time_to_eruption"]

    # Ensemble prediction
    val_preds = np.zeros(len(X_val))
    for model in models:
        # predict handles loading best iteration automatically if passed
        val_preds += model.predict(X_val, num_iteration=model.best_iteration)
    val_preds /= len(models)

    # Calculate Metric
    mae = mean_absolute_error(y_val, val_preds)
    print(f"Final Validation Metric: {mae}")

    # 5. Failure Analysis
    print("Running failure analysis...")
    errors = np.abs(y_val - val_preds)

    # Calculate correlation between absolute error and features
    correlations = {}
    for col in X_val.columns:
        # Handle potential constant columns or NaNs in correlation calculation
        try:
            corr = np.corrcoef(errors, X_val[col])[0, 1]
            if np.isnan(corr):
                corr = 0.0
            correlations[col] = corr
        except Exception:
            correlations[col] = 0.0

    # Sort by absolute correlation strength
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features correlated with Error:")
    for feature, corr in sorted_corrs[:10]:
        print(f"{feature}: {corr:.4f}")

    # 6. Submission
    threshold = 2617304.0647319085
    if mae < threshold:
        print(
            f"Validation metric {mae} is below threshold {threshold}. Generating submission..."
        )

        test_meta_path = os.path.join(config.METADATA_DIR, "test.csv")
        test_df = data_processor.process_set(
            metadata_path=test_meta_path,
            cache_name="test_features.parquet",
            load_cached_data=True,
        )

        if not test_df.empty:
            trainer.generate_submission(models, test_df)
        else:
            print("Error: Test DataFrame is empty.")
    else:
        print(
            f"Validation metric {mae} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
