import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from library.config import Config
from library.data_loader import generate_dataset
from library.model_handler import LGBMRegressorWrapper


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Configuration
    cfg = Config()
    set_seed(cfg.SEED)

    # 2. Data Loading
    # We load both the provided train and validation sets.
    # According to the strategy, we will merge them and create a custom 90/10 split
    # to maximize the training data available for the single model.
    print("--- Loading Labeled Data ---")
    X_train_orig, y_train_orig, _ = generate_dataset(
        cfg.TRAIN_METADATA, cfg, load_cached_data=True, dataset_name="train"
    )
    X_val_orig, y_val_orig, _ = generate_dataset(
        cfg.VAL_METADATA, cfg, load_cached_data=True, dataset_name="val"
    )

    # Merge datasets
    X_full = pd.concat([X_train_orig, X_val_orig], axis=0, ignore_index=True)
    y_full = pd.concat([y_train_orig, y_val_orig], axis=0, ignore_index=True)

    # Perform 90/10 Split
    print("--- Performing 90/10 Train-Validation Split ---")
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.10, random_state=cfg.SEED, shuffle=True
    )
    print(f"Train Set Shape: {X_train.shape}")
    print(f"Val Set Shape:   {X_val.shape}")

    # 3. Model Training
    print("--- Initializing and Training Model ---")
    model_wrapper = LGBMRegressorWrapper(cfg)
    model_wrapper.fit(X_train, y_train, X_val, y_val)

    # 4. Validation Evaluation
    print("--- Evaluating on Validation Split ---")
    val_preds = model_wrapper.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)

    # Required Output
    print(f"Final Validation Metric: {mae}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute errors
    errors = np.abs(y_val - val_preds)
    error_series = pd.Series(errors, index=X_val.index, name="abs_error")

    # Compute correlation between features and the error magnitude
    # We look for features that are strongly correlated with high errors
    correlations = X_val.corrwith(error_series).sort_values(ascending=False)

    print("Top 5 Features positively correlated with Error (High Value -> High Error):")
    print(correlations.head(5))

    print(
        "\nTop 5 Features negatively correlated with Error (Low Value -> High Error):"
    )
    print(correlations.tail(5))

    # 6. Submission Generation
    # Threshold defined in the task
    threshold = 2617304.0647319085

    if mae < threshold:
        print(
            f"\nValidation MAE ({mae}) meets the threshold ({threshold}). Proceeding to submission."
        )

        print("--- Loading Test Data ---")
        X_test, _, test_ids = generate_dataset(
            cfg.TEST_METADATA, cfg, load_cached_data=True, dataset_name="test"
        )

        print("--- Generating Submission ---")
        model_wrapper.generate_submission(X_test, test_ids)

    else:
        print(
            f"\nValidation MAE ({mae}) does not meet the threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
