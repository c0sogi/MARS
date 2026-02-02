import os
import pandas as pd
import numpy as np
import library.config as config
import library.data_loader as data_loader
import library.model_trainer as model_trainer


def run():
    # Set random seeds for reproducibility
    np.random.seed(config.SEED)

    # ==========================================
    # 1. Data Loading & Feature Engineering
    # ==========================================
    print("Loading Training Data...")
    # build_dataset handles loading raw CSVs, extracting features (Time/Freq/Window),
    # and caching the result to parquet.
    train_df = data_loader.build_dataset("train", load_cached_data=True)

    print("Loading Validation Data...")
    val_df = data_loader.build_dataset("val", load_cached_data=True)

    # ==========================================
    # 2. Prepare Features and Targets
    # ==========================================
    print("Preparing Feature Matrices...")
    # Split the dataframes into X (features) and y (target)
    X_train, y_train = data_loader.prepare_features_target(train_df)
    X_val, y_val = data_loader.prepare_features_target(val_df)

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("Training Model...")
    # We use a slightly reduced number of estimators (5000) compared to the config (10000)
    # to ensure the baseline runs within the time limit while still converging.
    model, val_mae = model_trainer.train_regressor(
        X_train, y_train, X_val, y_val, n_estimators=5000
    )

    # ==========================================
    # 4. Validation Assessment
    # ==========================================
    # Required output format for the metric
    print(f"Final Validation Metric: {val_mae}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")
    # Generate predictions on the validation set to analyze errors
    val_preds = model_trainer.predict(model, X_val)
    errors = np.abs(y_val - val_preds)

    # Create a Series for the errors to compute correlations
    error_series = pd.Series(errors, index=X_val.index, name="abs_error")

    # Compute correlation between the magnitude of error and feature values
    # This helps identify if specific signal characteristics (e.g., high noise, specific frequencies)
    # lead to higher prediction errors.
    correlations = X_val.corrwith(error_series).abs().sort_values(ascending=False)

    print("Top 5 Features correlated with Absolute Error:")
    print(correlations.head(5))

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    # Threshold defined in the task description
    THRESHOLD = 4164831.55

    if val_mae < THRESHOLD:
        print(
            f"\nValidation MAE ({val_mae}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data (Features are extracted and cached if not present)
        print("Loading Test Data...")
        test_df = data_loader.build_dataset("test", load_cached_data=True)

        # Generate and save submission file
        model_trainer.generate_submission(model, test_df)
    else:
        print(
            f"\nValidation MAE ({val_mae}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run()
