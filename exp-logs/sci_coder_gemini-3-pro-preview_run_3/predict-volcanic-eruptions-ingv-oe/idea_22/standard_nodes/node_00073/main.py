import os
import sys
import numpy as np
import pandas as pd
import random
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Import from the provided library files
from library.config import Config
from library.model import ModelTrainer
from library.dataset import VolcanoDataset
from library.utils import setup_logger


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and OS environments.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup and Initialization
    logger = setup_logger("RunFile")
    set_seed(Config.SEED)
    logger.info("Initializing pipeline...")

    # 2. Model Training
    # We initialize the ModelTrainer which encapsulates the LightGBM model and data loading.
    # We use load_cached_data=True to utilize any pre-computed features in ./working.
    # We do not limit the training data (limit=None) to ensure we achieve the best possible MAE
    # to beat the strict threshold, as feature extraction is parallelized and efficient.
    trainer = ModelTrainer()
    model = trainer.train(load_cached_data=True, limit=None)

    # 3. Validation & Metric Calculation
    # We manually retrieve the validation data to perform independent inference and analysis.
    dataset = VolcanoDataset()
    X_val, y_val = dataset.get_val_data(load_cached_data=True)

    # Perform inference on the validation set
    # LightGBM automatically handles the device (CPU/GPU) based on build and config.
    # For tabular inference on this scale (~800 rows), CPU is extremely fast and efficient.
    preds = model.predict(X_val, num_iteration=model.best_iteration)

    # Calculate and print the required metric
    mae = mean_absolute_error(y_val, preds)
    print(f"Final Validation Metric: {mae}")

    # 4. Failure Analysis
    # Identify systematic errors by correlating feature values with error magnitude.
    logger.info("Performing Failure Analysis on Validation Set...")

    # Calculate absolute errors
    errors = np.abs(y_val - preds)

    # Compute correlation between every feature and the error vector
    # This helps identify if specific sensor behaviors (e.g., high variance) lead to higher errors.
    error_correlations = X_val.corrwith(pd.Series(errors, index=X_val.index))

    # Sort by absolute correlation strength
    top_correlations = error_correlations.abs().sort_values(ascending=False).head(5)

    print("\n--- Failure Analysis: Top 5 Features Correlated with Error Magnitude ---")
    for feature, corr_value in top_correlations.items():
        # Retrieve the original sign from the unsorted series
        sign = error_correlations[feature]
        print(f"{feature}: {sign:.4f}")
    print("------------------------------------------------------------------------\n")

    # 5. Conditional Submission
    # The submission file is generated only if the validation MAE meets the specified threshold.
    THRESHOLD = 2617304.0647319085

    if mae < THRESHOLD:
        logger.info(f"Validation MAE ({mae}) meets the threshold ({THRESHOLD}).")
        logger.info("Generating submission file...")
        trainer.generate_submission(load_cached_data=True)
    else:
        logger.warning(
            f"Validation MAE ({mae}) did NOT meet the threshold ({THRESHOLD})."
        )
        logger.warning("Submission generation skipped.")


if __name__ == "__main__":
    main()
