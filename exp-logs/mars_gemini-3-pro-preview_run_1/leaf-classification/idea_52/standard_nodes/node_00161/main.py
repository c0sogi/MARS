import os
import sys
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, custom_log_loss
from library.data import DataManager
from library.model import OASLinearDiscriminant


def run_pipeline():
    # 1. Setup
    print("Setting up environment...")
    set_seed(Config.SEED)

    # 2. Data Loading & Preprocessing
    # DataManager handles feature extraction (images + tabular) and preprocessing
    print("Initializing DataManager...")
    dm = DataManager()

    print("Loading datasets (Train, Val, Test)...")
    # load_cached_data=True uses pre-computed features if available in ./working
    (train_data, val_data, test_data) = dm.load_all_data(load_cached_data=True)

    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data
    X_test, ids_test = test_data

    print(f"Data Loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 3. Model Training
    print("Training OAS Linear Discriminant...")
    model = OASLinearDiscriminant()
    model.fit(X_train, y_train)

    # 4. Validation
    print("Evaluating on Validation set...")
    y_val_pred = model.predict_proba(X_val)

    # Compute Metric
    # custom_log_loss applies the specific rescaling and clipping required by the task
    val_loss = custom_log_loss(y_val, y_val_pred)

    # Print Metric in required format
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude per sample: -log(p_true)
    # Map true labels to indices
    le = model.le
    y_val_indices = le.transform(y_val)

    # Clip predictions to match metric logic (eps=1e-15)
    eps = 1e-15
    y_val_pred_clipped = np.clip(y_val_pred, eps, 1 - eps)

    # Get probability of the true class for each sample
    prob_true = y_val_pred_clipped[np.arange(len(y_val)), y_val_indices]

    # Error magnitude
    error_magnitudes = -np.log(prob_true)

    # Calculate Pearson correlation between each feature and the error magnitude
    n_features = X_val.shape[1]
    correlations = []

    # We iterate through features to find which are associated with higher error
    for i in range(n_features):
        feat_values = X_val[:, i]
        # Avoid warning if variance is 0
        if np.std(feat_values) > 0 and np.std(error_magnitudes) > 0:
            corr = np.corrcoef(feat_values, error_magnitudes)[0, 1]
        else:
            corr = 0.0
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for idx, corr in correlations[:5]:
        print(f"Feature {idx}: {corr:.6f}")

    # 6. Submission
    # Strict threshold from task description
    THRESHOLD = 3.3382359570696616e-14

    if val_loss < THRESHOLD:
        print(f"\nValidation metric ({val_loss}) meets threshold ({THRESHOLD}).")
        print("Generating submission for Test set...")

        y_test_pred = model.predict_proba(X_test)

        # Prepare DataFrame
        submission_df = pd.DataFrame(y_test_pred, columns=model.classes_)
        submission_df.insert(0, "id", ids_test)

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nValidation metric ({val_loss}) does NOT meet threshold ({THRESHOLD})."
        )
        print("Submission file will NOT be generated.")


if __name__ == "__main__":
    run_pipeline()
