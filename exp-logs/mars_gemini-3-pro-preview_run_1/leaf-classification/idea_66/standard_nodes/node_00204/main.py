import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import components from the provided library files
from library.config import SEED, SUBMISSION_DIR
from library.utils import (
    set_seed,
    calculate_log_loss,
    normalize_probabilities,
    clip_probabilities,
)
from library.preprocessing import SanitizedGroupPreprocessor
from library.model import FactorizedOASLDA


def main():
    # 1. Setup
    set_seed(SEED)
    print("Starting execution of runfile.py...")

    # 2. Data Loading and Preprocessing
    # The preprocessor handles loading metadata, extracting geometric features,
    # splitting into groups, sanitizing (variance threshold), and transforming (Yeo-Johnson).
    print("Initializing preprocessor...")
    preprocessor = SanitizedGroupPreprocessor()

    # Load data (utilizes cache if available to speed up runtime)
    data = preprocessor.process_and_cache(load_cached_data=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    test_ids = data["test_ids"]
    classes = data["classes"]

    print(f"Data loaded successfully.")
    print(f"Train samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(test_ids)}")

    # 3. Model Training
    print("\nInitializing Factorized OAS-LDA Model...")
    model = FactorizedOASLDA()

    print("Fitting model on training data...")
    model.fit(X_train, y_train)
    print("Model fitting complete.")

    # 4. Validation
    print("\nRunning inference on validation set...")
    y_pred_val = model.predict_proba(X_val)

    # Calculate Validation Metric (Multi-class Log Loss)
    # We pass labels=range(len(classes)) because y_val are integer indices 0..98
    val_loss = calculate_log_loss(y_val, y_pred_val, labels=list(range(len(classes))))

    # REQUIRED: Print the final validation metric
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate per-sample log loss
    # Normalize and clip probabilities to match the metric calculation
    y_pred_norm = normalize_probabilities(y_pred_val)
    y_pred_clipped = clip_probabilities(y_pred_norm)

    # Extract the probability assigned to the true class for each sample
    n_samples = len(y_val)
    # Advanced indexing to get p(y_true|x)
    true_class_probs = y_pred_clipped[np.arange(n_samples), y_val]
    # Loss = -log(p_true)
    sample_losses = -np.log(true_class_probs)

    # Flatten the feature dictionary into a single matrix for correlation analysis
    # We sort keys to ensure deterministic ordering of features
    feature_values = []
    feature_names = []
    sorted_groups = sorted(X_val.keys())

    for group in sorted_groups:
        group_data = X_val[group]
        n_feats = group_data.shape[1]
        feature_values.append(group_data)
        feature_names.extend([f"{group}_{i}" for i in range(n_feats)])

    if feature_values:
        X_val_flat = np.hstack(feature_values)

        # Calculate Pearson correlation between features and error magnitude
        correlations = []
        for i in range(X_val_flat.shape[1]):
            feat_vec = X_val_flat[:, i]
            # Check for constant features to avoid division by zero in correlation
            if np.std(feat_vec) < 1e-12:
                corr = 0.0
            else:
                corr, _ = pearsonr(feat_vec, sample_losses)
                if np.isnan(corr):
                    corr = 0.0
            correlations.append((feature_names[i], corr))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 5 Features correlated with Prediction Error:")
        for name, corr in correlations[:5]:
            print(f"  {name}: {corr:.4f}")
    else:
        print("No features available for failure analysis.")

    # 6. Submission Generation
    # Strict threshold check as required
    THRESHOLD = 3.058881515561734e-14

    if val_loss < THRESHOLD:
        print(f"\nValidation metric ({val_loss}) meets the threshold (< {THRESHOLD}).")
        print("Generating predictions for test set...")

        y_pred_test = model.predict_proba(X_test)

        # Create submission DataFrame
        submission_df = pd.DataFrame(y_pred_test, columns=classes)
        # Insert 'id' column at the beginning
        submission_df.insert(0, "id", test_ids)

        # Save to CSV
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nValidation metric ({val_loss}) does NOT meet the threshold (< {THRESHOLD})."
        )
        print("Submission file will NOT be generated.")


if __name__ == "__main__":
    main()
