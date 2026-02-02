import sys
import os
import numpy as np
import pandas as pd
import random
from sklearn.metrics import matthews_corrcoef

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.model_trainer import ModelTrainer
from library.metric_optimizer import MetricOptimizer


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def main():
    # 1. Configuration and Setup
    # Modify n_estimators to ensure the pipeline runs within the time limit
    # while relying on early_stopping_rounds for convergence.
    Config.XGB_PARAMS_STREAM_A["n_estimators"] = 2000
    Config.XGB_PARAMS_STREAM_B["n_estimators"] = 2000

    Config.setup()
    set_seed(Config.SEED)

    # 2. Feature Engineering
    # Load and process data for all modes (train, validation, test)
    # Using cached data if available to save time
    fe = FeatureEngineer()

    print("Processing Train Data...")
    train_data = fe.process_data(mode="train", load_cached_data=True)

    print("Processing Validation Data...")
    val_data = fe.process_data(mode="validation", load_cached_data=True)

    print("Processing Test Data...")
    test_data = fe.process_data(mode="test", load_cached_data=True)

    # 3. Model Training
    # Train the Dual-Stream GBDT models
    trainer = ModelTrainer()
    models = trainer.train(train_data, val_data)

    # 4. Threshold Optimization & Validation
    optimizer = MetricOptimizer()

    # Find optimal thresholds per stream based on validation performance
    thresholds = optimizer.optimize_thresholds(models, val_data)

    # Calculate Final Validation Metric (Global MCC)
    val_preds_list = []
    val_truth_list = []

    for stream in ["stream_a", "stream_b"]:
        if (
            stream in models
            and models[stream] is not None
            and len(val_data[stream]["X"]) > 0
        ):
            X_val = val_data[stream]["X"]
            y_val = val_data[stream]["y"]
            model = models[stream]
            thresh = thresholds[stream]

            # Predict probabilities and apply threshold
            probs = model.predict_proba(X_val)[:, 1]
            preds = (probs >= thresh).astype(int)

            val_preds_list.append(preds)
            val_truth_list.append(y_val)

    if val_preds_list:
        all_val_preds = np.concatenate(val_preds_list)
        all_val_truth = np.concatenate(val_truth_list)
        final_mcc = matthews_corrcoef(all_val_truth, all_val_preds)
    else:
        final_mcc = 0.0

    print(f"Final Validation Metric: {final_mcc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    for stream in ["stream_a", "stream_b"]:
        if (
            stream in models
            and models[stream] is not None
            and len(val_data[stream]["X"]) > 0
        ):
            print(f"Analyzing systematic errors for {stream}...")
            X_val = val_data[stream]["X"]
            y_val = val_data[stream]["y"]
            model = models[stream]

            # Calculate Error Magnitude (Residuals)
            probs = model.predict_proba(X_val)[:, 1]
            errors = np.abs(y_val - probs)

            # Calculate correlation between features and error magnitude
            # X_val is a DataFrame, errors is a numpy array
            correlations = X_val.corrwith(pd.Series(errors, index=X_val.index))

            # Identify top 5 features associated with model error
            top_corr = correlations.abs().sort_values(ascending=False).head(5)
            print(f"Top 5 features correlated with error in {stream}:")
            print(top_corr)

    # 6. Submission Generation
    THRESHOLD_SCORE = 0.7008

    if final_mcc > THRESHOLD_SCORE:
        print(
            f"\nValidation metric ({final_mcc}) meets threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        # Generate raw probability predictions for the test set
        predictions_df = trainer.predict(models, test_data)

        # Apply optimized thresholds and save submission file
        optimizer.generate_submission(predictions_df, thresholds)

    else:
        print(
            f"\nValidation metric ({final_mcc}) does not meet threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
