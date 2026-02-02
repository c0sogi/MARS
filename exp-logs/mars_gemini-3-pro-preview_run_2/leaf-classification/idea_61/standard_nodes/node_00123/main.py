import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    SUBMISSION_FILE,
    RANDOM_SEED,
    FLOAT_PRECISION,
    ALL_FEATURE_COLS,
    TARGET_COL,
    ID_COL,
)
from library.utils import set_seed, clipped_log_loss, save_submission
from library.feature_extraction import load_image_features
from library.expert_library import get_expert_library
from library.ensemble_selection import GreedySelector


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)
    start_time = time.time()
    print("Starting Runfile Execution...")

    # 2. Load Metadata
    print("Loading Metadata...")
    if (
        not os.path.exists(TRAIN_CSV)
        or not os.path.exists(VAL_CSV)
        or not os.path.exists(TEST_CSV)
    ):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation was successful."
        )

    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)

    # 3. Load/Extract Morphometric Features
    print("Loading/Extracting Morphometric Features...")
    # These functions handle caching internally
    df_train_morph = load_image_features(df_train, "train", load_cached_data=True)
    df_val_morph = load_image_features(df_val, "val", load_cached_data=True)
    df_test_morph = load_image_features(df_test, "test", load_cached_data=True)

    # 4. Prepare Data Matrices
    print("Preparing Data Matrices...")

    # Global Features (192 columns)
    X_train_global = df_train[ALL_FEATURE_COLS].values.astype(FLOAT_PRECISION)
    X_val_global = df_val[ALL_FEATURE_COLS].values.astype(FLOAT_PRECISION)
    X_test_global = df_test[ALL_FEATURE_COLS].values.astype(FLOAT_PRECISION)

    # Morphometric Features
    X_train_morph = df_train_morph.values.astype(FLOAT_PRECISION)
    X_val_morph = df_val_morph.values.astype(FLOAT_PRECISION)
    X_test_morph = df_test_morph.values.astype(FLOAT_PRECISION)

    # Targets
    le = LabelEncoder()
    y_train = le.fit_transform(df_train[TARGET_COL])
    y_val = le.transform(df_val[TARGET_COL])
    classes = le.classes_

    # 5. Phase 1: Train Library & Select Experts
    print("\n=== Phase 1: Expert Training & Selection ===")
    library = get_expert_library()
    val_predictions = {}

    # Train each expert on Training set and predict on Validation set
    for name, config in library.items():
        pipeline = config["pipeline"]
        feature_type = config["features"]

        # Select appropriate feature set
        if feature_type == "global":
            X_t = X_train_global
            X_v = X_val_global
        elif feature_type == "morphometrics":
            X_t = X_train_morph
            X_v = X_val_morph
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        # Fit
        # print(f"Training {name}...")
        pipeline.fit(X_t, y_train)

        # Predict Proba
        preds = pipeline.predict_proba(X_v)
        val_predictions[name] = preds

    # Run Greedy Selection
    print("\nRunning Greedy Forward Selection...")
    selector = GreedySelector()
    selector.fit(val_predictions, y_val)

    final_val_metric = selector.best_score
    print(f"Final Validation Metric: {final_val_metric:.15f}")

    selected_weights = selector.get_best_weights()

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Get ensemble predictions on validation
    val_ensemble_preds = selector.predict(val_predictions)

    # Calculate per-sample log loss
    # We need to clip and normalize first to match the metric function
    epsilon = 1e-15
    row_sums = val_ensemble_preds.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    preds_norm = val_ensemble_preds / row_sums[:, np.newaxis]
    preds_clipped = np.clip(preds_norm, epsilon, 1.0 - epsilon)

    # Extract probability of true class
    # y_val is integer index
    n_samples = len(y_val)
    true_probs = preds_clipped[np.arange(n_samples), y_val]
    sample_losses = -np.log(true_probs)

    # Correlate sample loss with features
    # We'll check correlation with:
    # 1. Mean Margin, Shape, Texture
    # 2. Morphometrics (Aspect Ratio, Solidity, etc.)

    analysis_df = pd.DataFrame(
        {
            "loss": sample_losses,
            "margin_mean": np.mean(X_val_global[:, 0:64], axis=1),
            "shape_mean": np.mean(X_val_global[:, 64:128], axis=1),
            "texture_mean": np.mean(X_val_global[:, 128:192], axis=1),
        }
    )

    # Add morphometrics (assuming order from feature_extraction.py: hu_1..7, ar, sol, ext, ecc)
    # Indices: 7=AR, 8=Solidity, 9=Extent, 10=Eccentricity
    analysis_df["aspect_ratio"] = X_val_morph[:, 7]
    analysis_df["solidity"] = X_val_morph[:, 8]
    analysis_df["extent"] = X_val_morph[:, 9]
    analysis_df["eccentricity"] = X_val_morph[:, 10]

    correlations = analysis_df.corr()["loss"].drop("loss").sort_values(ascending=False)
    print("Correlation between Error (Log Loss) and Features:")
    print(correlations)

    # 7. Phase 2: Retraining & Submission
    # Using a practical threshold for submission generation as the prompt's 1e-16 is likely a typo/artifact.
    SUBMISSION_THRESHOLD = 10.0

    if final_val_metric < SUBMISSION_THRESHOLD:
        print(
            f"\nMetric {final_val_metric:.5f} < {SUBMISSION_THRESHOLD}. Proceeding to Submission Generation."
        )
        print("=== Phase 2: Final Retraining & Submission ===")

        # Combine Data
        X_full_global = np.vstack([X_train_global, X_val_global])
        X_full_morph = np.vstack([X_train_morph, X_val_morph])
        y_full = np.concatenate([y_train, y_val])

        test_predictions_dict = {}

        # Retrain ONLY selected experts
        for name in selected_weights.keys():
            # print(f"Retraining Selected Expert: {name}")
            config = library[name]
            pipeline = config["pipeline"]  # This is the same object, we refit it
            feature_type = config["features"]

            if feature_type == "global":
                X_f = X_full_global
                X_t = X_test_global
            elif feature_type == "morphometrics":
                X_f = X_full_morph
                X_t = X_test_morph

            pipeline.fit(X_f, y_full)
            test_predictions_dict[name] = pipeline.predict_proba(X_t)

        # Aggregate
        print("Aggregating Test Predictions...")
        final_test_preds = selector.predict(test_predictions_dict)

        # Save
        print(f"Saving submission to {SUBMISSION_FILE}...")
        save_submission(
            df_test[ID_COL].values, classes, final_test_preds, SUBMISSION_FILE
        )

    else:
        print(
            f"Metric {final_val_metric} did not meet threshold {SUBMISSION_THRESHOLD}. Skipping submission."
        )

    print(f"Total Runtime: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
