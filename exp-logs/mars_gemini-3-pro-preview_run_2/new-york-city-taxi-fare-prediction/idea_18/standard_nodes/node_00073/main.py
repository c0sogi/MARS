import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import from provided library files
from library.config import Config
from library.data_loader import load_training_data, load_validation_data, load_test_data
from library.feature_engineer import FeatureEngineer
from library.model_trainer import XGBTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Initialization
    set_seed(Config.RANDOM_SEED)
    print(
        "Starting Multi-Moment Hierarchical Dual-Hygiene Gradient Boosting Pipeline..."
    )

    # 2. Data Loading
    # Load training data (Wisdom + Learner)
    # Wisdom: Used for generating stats (Mean/Std/Count)
    # Learner: Used for training the Gradient Boosting model
    print("\n=== Data Loading ===")
    wisdom_df, learner_df = load_training_data(load_cached_data=True)
    val_df = load_validation_data(load_cached_data=True)
    test_df = load_test_data(load_cached_data=True)

    # 3. Feature Engineering
    print("\n=== Feature Engineering ===")
    fe = FeatureEngineer()

    # Process Training Data
    # This fits the stats engine on wisdom_df and transforms learner_df using K-Fold subtraction
    train_df_processed = fe.process_train_data(
        wisdom_df=wisdom_df, learner_df=learner_df, load_cached_data=True
    )

    # Process Validation Data
    # Uses global stats from wisdom_df (no subtraction)
    val_df_processed = fe.process_validation_data(
        val_df=val_df, wisdom_df=wisdom_df, load_cached_data=True
    )

    # Process Test Data
    # Uses global stats from wisdom_df (no subtraction)
    test_df_processed = fe.process_test_data(
        test_df=test_df, wisdom_df=wisdom_df, load_cached_data=True
    )

    # 4. Model Training
    print("\n=== Model Training ===")
    trainer = XGBTrainer()

    # Fit the model
    trainer.fit(train_df_processed, val_df_processed)

    # Save the model
    model_path = os.path.join(Config.WORKING_DIR, "xgb_model.json")
    trainer.save_model(model_path)

    # 5. Evaluation
    print("\n=== Evaluation ===")
    # Compute RMSE on the full validation set
    val_rmse = trainer.evaluate(val_df_processed)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_rmse}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate residuals
    X_val, y_val = trainer._prepare_data(val_df_processed, is_training=True)
    y_pred = trainer.predict(val_df_processed)
    residuals = np.abs(y_val - y_pred)

    # Create analysis dataframe
    analysis_df = X_val.copy()
    analysis_df["abs_error"] = residuals

    # Compute correlations with error
    # We want to see which features correlate most with high error
    correlations = (
        analysis_df.corrwith(analysis_df["abs_error"])
        .abs()
        .sort_values(ascending=False)
    )

    print("Top 10 features correlated with Absolute Error:")
    print(correlations.head(10))

    # 7. Submission Generation
    # Threshold check as per requirements
    THRESHOLD = 3.438959912830025

    if val_rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({val_rmse}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission_file(test_df_processed)
    else:
        print(
            f"\nValidation RMSE ({val_rmse}) did not meet threshold ({THRESHOLD}). Skipping submission generation."
        )

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
