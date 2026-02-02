import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data_pipeline import DataPipeline
from library.model_factory import train_laplace_solver, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Initialize Data Pipeline
    pipeline = DataPipeline()

    # 3. Process Datasets
    # We use load_cached_data=True to utilize any pre-computed features in ./working
    print("--- Processing Training Data ---")
    train_data = pipeline.process_dataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        split_name="train",
        load_cached_data=True,
        is_training=True,
    )

    print("--- Processing Validation Data ---")
    val_data = pipeline.process_dataset(
        metadata_path=Config.VAL_METADATA_PATH,
        split_name="val",
        load_cached_data=True,
        is_training=False,
    )

    # 4. Train Model
    # The linear solvers (QuantileRegressor and ElasticNet) are computationally efficient
    # and suitable for a fast baseline without aggressive subsampling on this dataset size.
    print("--- Training Laplace Solver ---")
    model = train_laplace_solver(train_data, val_data)

    # 5. High-Precision Evaluation
    X_fvc_val = val_data["X_fvc"]
    X_unc_val = val_data["X_unc"]
    y_val = val_data["y"]

    val_score = model.evaluate(X_fvc_val, X_unc_val, y_val)
    # Strictly required output format
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Predict on validation set to get errors
    fvc_pred, sigma_pred = model.predict(X_fvc_val, X_unc_val)
    abs_error = np.abs(y_val - fvc_pred)

    # Load raw metadata to correlate errors with interpretable features
    try:
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)

        # Add analysis columns
        df_val["AbsError"] = abs_error
        df_val["PredictedSigma"] = sigma_pred

        # Select numerical features for correlation
        # We focus on Weeks, Age, and Percent as primary clinical indicators
        analysis_features = ["Weeks", "Age", "Percent", "AbsError"]

        # Compute correlation matrix
        corr_matrix = df_val[analysis_features].corr()
        error_corr = corr_matrix["AbsError"].sort_values(ascending=False)

        print("Correlation between Absolute Error and Input Features:")
        print(error_corr.drop("AbsError"))  # Drop self-correlation

    except Exception as e:
        print(f"Failure analysis could not be completed: {e}")

    # 7. Conditional Submission
    # Threshold defined in requirements
    THRESHOLD = -6.805292148096688

    if val_score > THRESHOLD:
        print(
            f"\nValidation metric {val_score} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        print("--- Processing Test Data ---")
        test_data = pipeline.process_dataset(
            metadata_path=Config.TEST_METADATA_PATH,
            split_name="test",
            load_cached_data=True,
            is_training=False,
        )

        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        generate_submission(
            model=model,
            data_dict_test=test_data,
            test_metadata_path=Config.TEST_METADATA_PATH,
            output_path=submission_path,
        )
    else:
        print(
            f"\nValidation metric {val_score} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
