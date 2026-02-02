import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Import provided library components
from library.config import Config
from library.data_factory import DataFactory
from library.model_trainer import ResidualXGBRegressor


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup and Configuration
    Config.setup()
    set_seed(Config.RANDOM_SEED)

    # Optimization for Fast Baseline:
    # Limit training size and rounds to ensure execution within 2 hours
    # while maintaining enough data for the hierarchical priors to work.
    Config.TRAIN_SUBSAMPLE_SIZE = 2_000_000
    Config.NUM_BOOST_ROUNDS = 1000

    # 2. Model Training
    print("Initializing and training model...")
    regressor = ResidualXGBRegressor()
    # load_cached_data=True ensures we use pre-processed data if available in ./working
    regressor.train(load_cached_data=True)

    # 3. Validation Inference
    print("Loading validation data for evaluation...")
    # DataFactory handles loading and physics-consistent filtering
    val_df = DataFactory.load_val_data(load_cached_data=True)

    print("Featurizing validation data...")
    # We must use the feature_engineer from the trained regressor to ensure
    # the Global Knowledge Base (priors) matches what the model learned.
    if regressor.feature_engineer is None:
        raise RuntimeError("Model training failed or Feature Engineer not initialized.")

    val_feat = regressor.feature_engineer.process(val_df, is_training=False)

    # Prepare DMatrix for inference
    # Ensure we use exactly the same features as training
    features = regressor.features
    print(f"Inference using {len(features)} features.")

    # Create DMatrix (XGBoost handles device placement based on training params)
    dval = xgb.DMatrix(val_feat[features])

    print("Predicting on validation set...")
    # Predict the residual (Error from the Base Margin)
    pred_residual = regressor.model.predict(dval)

    # Final Prediction = Base Margin + Predicted Residual
    pred_fare = val_feat["base_margin"].values + pred_residual

    # Post-processing: Clamp to minimum fare
    pred_fare = np.maximum(pred_fare, Config.MIN_FARE_PREDICTION)

    # 4. Metric Calculation
    y_true = val_feat["fare_amount"].values
    rmse = np.sqrt(mean_squared_error(y_true, pred_fare))

    # Print strictly formatted metric
    print(f"Final Validation Metric: {rmse}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error magnitude
    errors = np.abs(y_true - pred_fare)

    # Create analysis dataframe
    analysis_df = val_feat[features].copy()
    analysis_df["error_magnitude"] = errors

    # Calculate correlations
    print("Calculating feature correlations with error magnitude...")
    correlations = analysis_df.corrwith(analysis_df["error_magnitude"])
    correlations = correlations.drop("error_magnitude", errors="ignore")

    # Get top 5 correlated features
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    print("Top 5 Features correlated with Error Magnitude:")
    for feature, val in top_correlations.items():
        # Print feature and its original correlation sign
        print(f"{feature}: {correlations[feature]:.4f}")

    # 6. Submission Logic
    THRESHOLD = 3.5069767944123895
    print(f"\nChecking submission criteria (RMSE < {THRESHOLD})...")

    if rmse < THRESHOLD:
        print(f"Validation RMSE {rmse} meets criteria. Generating submission...")
        regressor.generate_submission(load_cached_data=True)
        print("Submission file generated successfully.")
    else:
        print(f"Validation RMSE {rmse} does not meet criteria. Submission skipped.")


if __name__ == "__main__":
    main()
