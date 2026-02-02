import os
import sys
import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr

# Suppress warnings for clean execution
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.data_processor import DataProcessor
from library.expert_models import get_expert_pool
from library.ensemble_selector import GreedySelector


def set_seed(seed=42):
    """
    Sets random seeds for reproducibility across numpy and python.
    """
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(X, y_true, y_pred, feature_names):
    """
    Analyzes the correlation between prediction error and input features.
    Prints the top 5 features most correlated with high error.
    """
    # Calculate error per sample: -log(p_true_class)
    n_samples = len(y_true)
    errors = np.zeros(n_samples)

    # Clip and normalize probabilities for numerical stability
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    row_sums = y_pred_clipped.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred_clipped / row_sums

    for i in range(n_samples):
        true_class = y_true[i]
        prob = y_pred_norm[i, true_class]
        errors[i] = -np.log(prob)

    print("-" * 30)
    print("Failure Analysis (Correlation between Error and Features):")

    correlations = []
    # Calculate Pearson correlation for each feature
    for idx, feat_name in enumerate(feature_names):
        if idx >= X.shape[1]:
            break
        feat_values = X[:, idx]
        corr, _ = pearsonr(feat_values, errors)
        if np.isnan(corr):
            corr = 0.0
        correlations.append((feat_name, corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # Print top 5
    for name, corr in correlations[:5]:
        print(f"Feature: {name}, Correlation: {corr:.4f}")
    print("-" * 30)


def main():
    # 1. Initialization
    set_seed(Config.RANDOM_SEED)

    # 2. Data Loading
    processor = DataProcessor()
    print("Loading and processing data...")
    data = processor.load_and_process_data(load_cached_data=False)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    test_ids = data["test_ids"]
    classes = data["classes"]

    # 3. Phase 1: Expert Training & Selection
    print("Phase 1: Training experts and selecting ensemble...")
    experts = get_expert_pool()
    val_predictions = {}

    # Train each expert on the training split
    for name, model in experts.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        val_predictions[name] = model.predict_proba(X_val)

    # Select best ensemble composition
    selector = GreedySelector()
    all_labels = np.arange(len(classes))  # Ensure log_loss knows all potential classes
    selector.fit(val_predictions, y_val, labels=all_labels)

    # 4. Validation Assessment
    final_val_preds = selector.predict(val_predictions)

    # Calculate metric using the selector's internal clipped log loss method
    val_metric = selector._calculate_log_loss(y_val, final_val_preds, labels=all_labels)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    perform_failure_analysis(X_val, y_val, final_val_preds, processor.feature_cols)

    # 6. Phase 2: Submission
    # The prompt specifies a threshold of 3.3768e-06.
    THRESHOLD = 3.3768e-06

    if val_metric < THRESHOLD:
        print(
            f"Validation metric {val_metric:.9f} meets threshold {THRESHOLD}. Proceeding to submission."
        )

        # Combine Train and Validation sets for final training
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        test_predictions = {}
        selected_expert_names = list(selector.weights.keys())

        print("Phase 2: Retraining selected experts on full dataset...")
        for name in selected_expert_names:
            print(f"Retraining {name}...")
            model = experts[name]
            # Retrain on full data
            model.fit(X_full, y_full)
            # Predict on test data
            test_predictions[name] = model.predict_proba(X_test)

        # Aggregate predictions using learned weights
        final_test_preds = selector.predict(test_predictions)

        # Format Submission
        submission_df = pd.DataFrame(final_test_preds, columns=classes)
        submission_df.insert(0, "id", test_ids)

        # Save
        Config.ensure_directories()
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric {val_metric:.9f} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
