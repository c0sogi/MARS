import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

# Import provided library modules
from library.config import Config
from library.data_manager import get_train_data, get_val_data, get_test_data
from library.preprocessor import HighPrecisionPipeline
from library.model import IntegralInertialDiscriminant


def main():
    # 1. Setup and Configuration
    np.random.seed(Config.SEED)
    pd.set_option("mode.chained_assignment", None)

    print(
        "Starting execution of Integral-Inertial High-Precision OAS Discriminant pipeline..."
    )

    # 2. Data Loading
    # We use the data_manager to handle caching and parallel extraction
    print("Loading datasets...")
    train_df = get_train_data(load_cached_data=True)
    val_df = get_val_data(load_cached_data=True)
    test_df = get_test_data(load_cached_data=True)

    # 3. Data Splitting and Cleaning
    def prepare_data(df, is_test=False):
        # Remove excluded features defined in Config
        cols_to_drop = [c for c in Config.EXCLUDED_FEATURES if c in df.columns]
        df_clean = df.drop(columns=cols_to_drop)

        ids = df_clean["id"].values

        if is_test:
            y = None
            # Drop ID column to get X
            X = df_clean.drop(columns=["id"])
        else:
            y = df_clean["species"].values
            # Drop ID and Target columns to get X
            X = df_clean.drop(columns=["id", "species"])

        return X, y, ids

    print("Preparing feature matrices...")
    X_train_raw, y_train, ids_train = prepare_data(train_df, is_test=False)
    X_val_raw, y_val, ids_val = prepare_data(val_df, is_test=False)
    X_test_raw, _, ids_test = prepare_data(test_df, is_test=True)

    # 4. Preprocessing
    # Initialize and fit the HighPrecisionPipeline (Float64, PowerTransform, Scaling)
    print("Fitting HighPrecisionPipeline on training data...")
    pipeline = HighPrecisionPipeline()

    # Fit on Train, Transform Train
    X_train = pipeline.fit_transform(X_train_raw)

    # Transform Validation and Test
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    # 5. Model Training
    print(f"Training IntegralInertialDiscriminant on {len(X_train)} samples...")
    model = IntegralInertialDiscriminant()
    model.fit(X_train, y_train)

    # 6. Validation
    print("Performing validation inference...")
    val_probs = model.predict_proba(X_val)

    # Calculate Multi-class Log Loss
    # labels=model.classes_ ensures correct column mapping
    score = log_loss(y_val, val_probs, labels=model.classes_)

    # Print metric in required format
    print(f"Final Validation Metric: {score}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Encode validation labels to indices for lookup
    le = LabelEncoder()
    le.classes_ = model.classes_
    y_val_idx = le.transform(y_val)

    # Calculate per-sample log loss
    # Clip probabilities to avoid log(0)
    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)

    # Extract probability assigned to the true class
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_val_idx]
    sample_losses = -np.log(true_class_probs)

    # Create a DataFrame for correlation analysis using raw features (for interpretability)
    analysis_df = X_val_raw.copy()
    analysis_df["loss_magnitude"] = sample_losses

    # Compute correlation of features with error magnitude
    # Select only numeric columns
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
    correlations = (
        analysis_df[numeric_cols].corr()["loss_magnitude"].drop("loss_magnitude")
    )

    # Display top correlations
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)
    print("Top features correlated with prediction error:")
    print(top_correlations)

    # 8. Submission Generation
    print("\n--- Submission Generation ---")

    # Threshold from instructions (likely a typo for 3.338..., but kept for reference)
    threshold = 3.3382359570696616e-14

    # Generate predictions
    test_probs = model.predict_proba(X_test)

    # Format submission
    submission_df = pd.DataFrame(test_probs, columns=model.classes_)
    submission_df.insert(0, "id", ids_test)

    # Save submission
    # We save unconditionally to fulfill the "You must submit a csv file" requirement,
    # but we log the check against the strict threshold.
    if score < threshold:
        print(f"Validation score meets the strict threshold ({score} < {threshold}).")
    else:
        print(
            f"Validation score ({score}) did not meet the strict threshold ({threshold}). Saving submission regardless to ensure task completion."
        )

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
