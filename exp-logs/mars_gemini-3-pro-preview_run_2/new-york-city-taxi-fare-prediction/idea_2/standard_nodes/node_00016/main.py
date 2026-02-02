import os
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import random

# Import provided library components
from library.config import TRAIN_CONFIG, XGB_PARAMS, PATH_CONFIG, SEED
from library.model_trainer import XGBTrainer
from library.evaluator import ModelEvaluator
from library.data_processor import TaxiDataProcessor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class FastXGBTrainer(XGBTrainer):
    """
    Subclass of XGBTrainer to implement data subsampling for faster training
    without modifying the original library code.
    """

    def train(self, load_cached_data=True, sample_size=2000000):
        """
        Trains the XGBoost model using a subset of the training data.
        """
        # Load processed data (loads full dataset into memory, then we sample)
        # Note: 220GB RAM is sufficient to hold the full dataset before sampling.
        train_df = self.processor.get_processed_data(
            "train", load_cached_data=load_cached_data
        )
        val_df = self.processor.get_processed_data(
            "val", load_cached_data=load_cached_data
        )

        # Subsample training data for speed
        if len(train_df) > sample_size:
            train_df = train_df.sample(n=sample_size, random_state=SEED)

        # Prepare features and targets
        drop_cols = ["key", "fare_amount"]

        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df["fare_amount"]

        X_val = val_df.drop(columns=drop_cols)
        y_val = val_df["fare_amount"]

        # Configure XGBoost
        params = XGB_PARAMS.copy()
        params["n_estimators"] = TRAIN_CONFIG["num_boost_round"]
        params["early_stopping_rounds"] = TRAIN_CONFIG["early_stopping_rounds"]

        # Initialize model
        self.model = xgb.XGBRegressor(**params)

        # Train
        # print(f"Starting training with {len(X_train)} samples...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,  # Silent training as requested
        )

        # Save the model
        model_path = PATH_CONFIG["model_save_path"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_model(model_path)
        # print(f"Model saved to {model_path}")


def perform_failure_analysis(evaluator, sample_size=500000):
    """
    Analyzes model failures by correlating error magnitude with features.
    """
    # Load validation data
    val_df = evaluator.processor.get_processed_data("val", load_cached_data=True)

    # Sample for analysis speed if needed
    if len(val_df) > sample_size:
        val_df = val_df.sample(n=sample_size, random_state=SEED)

    target_col = "fare_amount"
    drop_cols = ["key", target_col]

    X_val = val_df.drop(columns=drop_cols)
    y_true = val_df[target_col].values

    # Predict
    raw_preds = evaluator.predict(X_val)
    final_preds = evaluator.post_process(raw_preds)

    # Calculate Error
    errors = np.abs(y_true - final_preds)

    # Create Analysis DataFrame
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = errors

    # Calculate Correlations
    correlations = analysis_df.corrwith(analysis_df["error_magnitude"])

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    # Sort by absolute correlation
    sorted_corr = correlations.abs().sort_values(ascending=False)
    for feature, corr_val in sorted_corr.items():
        if feature != "error_magnitude":
            print(f"{feature}: {correlations[feature]:.4f}")


def main():
    # 1. Setup
    set_seed(SEED)

    # Modify configuration for faster baseline execution
    # Limit boost rounds to ensure training completes quickly
    TRAIN_CONFIG["num_boost_round"] = 1000

    # 2. Train
    # Use custom trainer to limit training samples to 5 Million (approx 10% of data)
    # This balances speed and performance for a strong baseline.
    trainer = FastXGBTrainer()
    trainer.train(sample_size=5000000)

    # 3. Validation
    evaluator = ModelEvaluator()
    # Calculate metrics on the full validation set
    rmse = evaluator.calculate_metrics()
    print(f"Final Validation Metric: {rmse}")

    # 4. Failure Analysis
    perform_failure_analysis(evaluator)

    # 5. Submission
    # Threshold from task description
    THRESHOLD = 619.3649073538204

    if rmse < THRESHOLD:
        evaluator.generate_submission()
    else:
        print(
            f"Validation metric {rmse} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
