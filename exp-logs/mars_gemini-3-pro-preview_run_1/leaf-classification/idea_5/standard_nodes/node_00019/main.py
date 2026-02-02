import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder

# Import library modules
from library.config import Config
from library.data_loader import load_datasets
from library.preprocessor import preprocess_data
from library.model import FisherGaussianEnsemble
from library.evaluation import calculate_log_loss, create_submission_file


def run_pipeline():
    # 1. Set random seeds
    np.random.seed(Config.RANDOM_SEED)

    # 2. Load Data
    print("Loading datasets...")
    # Use cached data if available for speed
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids = load_datasets(
        load_cached_data=True
    )

    # 3. Preprocessing
    print("Preprocessing data...")
    # Apply Yeo-Johnson and Standard Scaling
    X_train, X_val, X_test = preprocess_data(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=True
    )

    # 4. Model Training
    print("Training Fisher-Gaussian Process Pipeline...")
    model = FisherGaussianEnsemble()
    model.fit(X_train, y_train)

    # 5. Validation
    print("Performing validation...")
    y_pred_proba_val = model.predict_proba(X_val)

    # Calculate and print metric
    val_loss = calculate_log_loss(y_val, y_pred_proba_val, model.classes_)
    print(f"Final Validation Metric: {val_loss}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Encode labels to indices to extract probability of true class
    le = LabelEncoder()
    le.fit(model.classes_)
    y_val_indices = le.transform(y_val)

    # Extract probability assigned to the true class
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    # Advanced indexing to get prob of true class for each sample
    probs_true = y_pred_proba_val[np.arange(len(y_val)), y_val_indices]
    probs_true = np.clip(probs_true, epsilon, 1.0)

    # Calculate per-sample log loss (error magnitude)
    sample_losses = -np.log(probs_true)

    # Calculate correlation between features and error
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vals = X_val[:, i]
        # Handle constant features to avoid warnings
        if np.std(feature_vals) < 1e-9:
            corr = 0.0
        else:
            corr, _ = pearsonr(feature_vals, sample_losses)
        correlations.append(corr)

    correlations = np.array(correlations)

    # Identify top features correlated with error
    # Positive correlation: Higher feature value -> Higher Error
    # Negative correlation: Higher feature value -> Lower Error
    top_pos_indices = np.argsort(correlations)[-5:][::-1]
    top_neg_indices = np.argsort(correlations)[:5]

    print("Top features associated with high error (Positive Correlation):")
    for idx in top_pos_indices:
        group_idx = idx // 64
        feat_offset = (idx % 64) + 1
        feat_name = f"{Config.FEATURE_GROUPS[group_idx]}{feat_offset}"
        print(f"  {feat_name}: {correlations[idx]:.4f}")

    print("Top features associated with low error (Negative Correlation):")
    for idx in top_neg_indices:
        group_idx = idx // 64
        feat_offset = (idx % 64) + 1
        feat_name = f"{Config.FEATURE_GROUPS[group_idx]}{feat_offset}"
        print(f"  {feat_name}: {correlations[idx]:.4f}")

    # 7. Submission Generation
    # Strict threshold check as per requirements
    THRESHOLD = 1.4705447816556679e-08

    if val_loss < THRESHOLD:
        print(
            f"\nValidation metric ({val_loss}) is lower than threshold ({THRESHOLD})."
        )
        create_submission_file(
            model, X_test, test_ids, output_path=Config.SUBMISSION_FILE
        )
    else:
        print(
            f"\nValidation metric ({val_loss}) is NOT lower than threshold ({THRESHOLD})."
        )
        print("Skipping submission generation as per strict instructions.")


if __name__ == "__main__":
    run_pipeline()
