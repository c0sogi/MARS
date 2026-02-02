import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Ensure library can be imported
sys.path.append(os.getcwd())

from library import config
from library import data_loader
from library import fe_bgp_model


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(config.RANDOM_SEED)

    # 2. Load Data
    # Using cached data for speed as per instructions
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, test_ids = data_loader.load_data(
        load_cached_data=True
    )

    # 3. Train on Training Set
    print("Training validation model...")
    model = fe_bgp_model.FisherEmbeddedBGP(random_state=config.RANDOM_SEED)
    model.fit(X_train, y_train)

    # 4. Validation Inference
    print("Performing validation inference...")
    val_probs = model.predict_proba(X_val)

    # Calculate Metric
    # Ensure labels provided to log_loss match the columns of val_probs
    metric = log_loss(y_val, val_probs, labels=model.classes_)
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    print("\nRunning Failure Analysis...")
    # Map string labels to column indices
    class_to_idx = {cls: i for i, cls in enumerate(model.classes_)}
    # Get integer indices for true labels
    y_val_indices = np.array([class_to_idx[y] for y in y_val])

    # Extract probability assigned to the true class
    # Use advanced indexing: row i, col y_val_indices[i]
    prob_true = val_probs[np.arange(len(y_val)), y_val_indices]

    # Calculate log loss per sample (clipped to avoid log(0))
    # Note: sklearn log_loss uses a specific clipping, we replicate logic for analysis
    prob_true_clipped = np.clip(prob_true, 1e-15, 1 - 1e-15)
    sample_losses = -np.log(prob_true_clipped)

    # Calculate correlation between features and error magnitude
    correlations = []
    for i in range(X_val.shape[1]):
        feature_vals = X_val[:, i]
        # Handle constant features to avoid warnings
        if np.std(feature_vals) == 0:
            corr = 0
        else:
            corr, _ = pearsonr(feature_vals, sample_losses)
        correlations.append(corr)

    correlations = np.array(correlations)
    # Get top 5 features positively correlated with error (high feature value -> high error)
    top_corr_indices = np.argsort(correlations)[::-1][:5]

    print("Top 5 features correlated with high prediction error:")
    for idx in top_corr_indices:
        # We don't have feature names in numpy array, but we know the order is sorted
        # We can construct the name if we loaded the dataframe, but here we just print index
        print(f"Feature Index {idx}: Correlation = {correlations[idx]:.4f}")

    # 6. Submission Logic
    THRESHOLD = 1.470544781593644e-08

    if metric < THRESHOLD:
        print(
            f"\nValidation metric ({metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Retrain on full data (Train + Val)
        print("Retraining on full dataset...")
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        global_model = fe_bgp_model.FisherEmbeddedBGP(random_state=config.RANDOM_SEED)
        global_model.fit(X_full, y_full)

        # Predict on Test
        print("Predicting on test set...")
        test_probs = global_model.predict_proba(X_test)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(test_probs, columns=global_model.classes_)
        submission_df.insert(0, config.ID_COL, test_ids)

        # Save
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(config.SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_FILE_PATH}")

    else:
        print(
            f"\nValidation metric ({metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
