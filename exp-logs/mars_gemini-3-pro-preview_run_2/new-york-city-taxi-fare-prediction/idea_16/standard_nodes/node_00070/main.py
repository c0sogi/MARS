import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Import provided library modules
from library.config import Config
from library.data_processor import prepare_datasets, load_data
from library.margin_logic import construct_train_margins, construct_test_margins
from library.feature_generator import process_features, prepare_dmatrix
from library.model_wrapper import XGBResiduaLearner


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Initialization
    set_seed(Config.SEED)
    print("Starting Hierarchical Residual Dual-Hygiene Pipeline...")

    # 2. Data Preparation
    # Load Wisdom (Strict Stats) and Learner (Loose Training) sets
    # This handles loading from cache if available
    wisdom_df, learner_df = prepare_datasets(load_cached_data=True)

    # Construct Margins for the Learner Set
    # Uses K-Fold subtraction to prevent leakage of the target into the margin
    learner_df = construct_train_margins(learner_df, wisdom_df, load_cached_data=True)

    # Feature Engineering for Learner Set
    learner_df = process_features(
        learner_df, cache_key="featurized_train", load_cached_data=True
    )

    # Define Feature Columns (exclude non-features)
    exclude_cols = {"key", "fare_amount", "pickup_datetime", "margin"}
    feature_cols = [c for c in learner_df.columns if c not in exclude_cols]
    print(f"Training with features: {feature_cols}")

    # 3. Model Training
    print("Preparing training data...")
    # Split Learner set for local validation (early stopping)
    train_df, local_val_df = train_test_split(
        learner_df, test_size=0.1, random_state=Config.SEED
    )

    # Create DMatrices (handling base_margin internally)
    dtrain = prepare_dmatrix(train_df, feature_cols, target_col="fare_amount")
    dval = prepare_dmatrix(local_val_df, feature_cols, target_col="fare_amount")

    # Initialize and Train Model
    model = XGBResiduaLearner()
    model.train(dtrain, dval)
    model.save("xgb_model.json")

    # Clear memory
    del learner_df, train_df, local_val_df, dtrain, dval
    import gc

    gc.collect()

    # 4. Official Validation
    print("Loading official validation set...")
    val_df = load_data(Config.VAL_DATA_PATH)

    # Apply Margin Logic (Inference Mode - Use full Global Stats)
    val_df = construct_test_margins(val_df, wisdom_df, load_cached_data=False)

    # Feature Engineering
    val_df = process_features(val_df, cache_key=None, load_cached_data=False)

    # Prepare DMatrix for Prediction
    dval_full = prepare_dmatrix(val_df, feature_cols, target_col="fare_amount")

    # Predict
    print("Predicting on validation set...")
    val_preds = model.predict(dval_full)

    # Calculate Metric
    val_rmse = np.sqrt(mean_squared_error(val_df["fare_amount"], val_preds))
    print(f"Final Validation Metric: {val_rmse}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    val_df["prediction"] = val_preds
    val_df["abs_error"] = np.abs(val_df["fare_amount"] - val_df["prediction"])

    # Calculate correlations with error
    # Select numerical columns only
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns
    correlations = {}
    for col in numeric_cols:
        if col not in ["abs_error", "prediction", "fare_amount", "margin"]:
            try:
                corr = val_df[col].corr(val_df["abs_error"])
                correlations[col] = corr
            except:
                pass

    # Sort and print top correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top Feature Correlations with Absolute Error:")
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val:.4f}")

    # 6. Submission
    THRESHOLD = 3.438959912830025
    if val_rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({val_rmse}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_df = load_data(Config.TEST_DATA_PATH)

        # Apply Margin Logic
        test_df = construct_test_margins(test_df, wisdom_df, load_cached_data=True)

        # Feature Engineering
        test_df = process_features(
            test_df, cache_key="featurized_test", load_cached_data=True
        )

        # Prepare DMatrix
        dtest = prepare_dmatrix(test_df, feature_cols)

        # Predict
        test_preds = model.predict(dtest)

        # Generate Submission File
        model.generate_submission(test_df, test_preds)
    else:
        print(
            f"\nValidation RMSE ({val_rmse}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
