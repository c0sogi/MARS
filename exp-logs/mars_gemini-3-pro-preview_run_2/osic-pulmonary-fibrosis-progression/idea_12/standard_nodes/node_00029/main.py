import sys
import os
import pandas as pd
import numpy as np

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, score_function, save_results
from library.data_manager import DataManager
from library.model_pipeline import DecoupledQuantileModel


def main():
    # 1. Configuration and Seeding
    seed_everything(Config.SEED)

    # 2. Initialize Data Manager
    dm = DataManager()

    # 3. Load Datasets
    # Using load_cached_data=True to utilize preprocessed features in ./working
    print("Loading Training Data...")
    train_data = dm.prepare_dataset("train", load_cached_data=True)

    print("Loading Validation Data...")
    val_data = dm.prepare_dataset("val", load_cached_data=True)

    # 4. Initialize and Train Model
    print("Initializing Model Pipeline...")
    model = DecoupledQuantileModel()

    print("Training Model...")
    # The model pipeline handles scaling, PCA, and regressor fitting
    model.fit(
        train_data["X_static"],
        train_data["weeks"],
        train_data["y"],
        train_data["base_weeks"],
    )

    # 5. Validation Inference
    print("Running Validation Inference...")
    val_preds, val_sigma = model.predict(
        val_data["X_static"], val_data["weeks"], val_data["base_weeks"]
    )

    # 6. Compute and Print Metric
    val_score = score_function(val_data["y"], val_preds, val_sigma)
    # Requirement: Print full precision
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Load metadata to get interpretable features for correlation
    val_meta_df = pd.read_csv(Config.VAL_META_PATH)

    # Calculate absolute error
    abs_error = np.abs(val_data["y"] - val_preds)

    # Create analysis dataframe
    # We use the numerical columns available in metadata
    analysis_df = val_meta_df[["Weeks", "Age", "Percent"]].copy()
    analysis_df["Abs_Error"] = abs_error
    analysis_df["Pred_Sigma"] = val_sigma
    analysis_df["Target_FVC"] = val_data["y"]

    # Compute correlations with Error
    correlations = analysis_df.corr()["Abs_Error"].sort_values(ascending=False)
    print("Correlation of features with Absolute Error:")
    print(correlations)

    # 8. Submission Logic
    THRESHOLD = -6.805292148096688

    if val_score > THRESHOLD:
        print(
            f"\nValidation score ({val_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        print("Loading Test Data...")
        test_data = dm.prepare_dataset("test", load_cached_data=True)

        print("Predicting on Test Set...")
        test_preds, test_sigma = model.predict(
            test_data["X_static"], test_data["weeks"], test_data["base_weeks"]
        )

        # Format Submission
        # Construct Patient_Week IDs
        patient_ids = test_data["patient_ids"]
        target_weeks = test_data["weeks"].astype(int)
        submission_ids = [
            f"{pid}_{week}" for pid, week in zip(patient_ids, target_weeks)
        ]

        df_sub = pd.DataFrame(
            {
                "Patient_Week": submission_ids,
                "FVC": test_preds,
                "Confidence": test_sigma,
            }
        )

        # Save
        save_results(df_sub, Config.SUBMISSION_FILE)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nValidation score ({val_score}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
