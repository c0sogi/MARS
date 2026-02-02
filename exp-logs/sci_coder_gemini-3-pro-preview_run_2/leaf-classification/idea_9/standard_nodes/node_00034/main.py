import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

# Import provided library modules
from library.data_loader import load_dataset
from library.ensemble_engine import StackingEnsemble
from library.utils import save_submission

# Set constants
RANDOM_SEED = 42
METRIC_THRESHOLD = 0.010054905410813797


def set_seed(seed):
    np.random.seed(seed)


def perform_failure_analysis(X, y, probs, feature_names=None):
    """
    Calculates per-sample log loss and correlates it with features.
    """
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)

    # Calculate per-sample log loss
    # Create one-hot encoding of true labels
    n_samples, n_classes = probs.shape
    # We assume y is integer encoded 0 to n_classes-1
    # Create index array for the true class
    rows = np.arange(n_samples)
    true_class_probs = probs_clipped[rows, y]
    sample_losses = -np.log(true_class_probs)

    print("\nFailure Analysis Report")
    print("-" * 30)
    print(f"Mean Loss: {np.mean(sample_losses):.6f}")
    print(f"Max Loss:  {np.max(sample_losses):.6f}")

    # Calculate correlation with features
    n_features = X.shape[1]
    correlations = []

    for i in range(n_features):
        # Handle constant features to avoid division by zero in correlation
        if np.std(X[:, i]) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(X[:, i], sample_losses)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Get top 5 positive and negative correlations
    top_pos_indices = np.argsort(correlations)[-5:][::-1]
    top_neg_indices = np.argsort(correlations)[:5]

    print("\nTop Features associated with High Error (Positive Corr):")
    for idx in top_pos_indices:
        feat_name = f"Feature_{idx}" if feature_names is None else feature_names[idx]
        print(f"  {feat_name}: {correlations[idx]:.4f}")

    print("\nTop Features associated with Low Error (Negative Corr):")
    for idx in top_neg_indices:
        feat_name = f"Feature_{idx}" if feature_names is None else feature_names[idx]
        print(f"  {feat_name}: {correlations[idx]:.4f}")


def main():
    set_seed(RANDOM_SEED)

    print("Initializing Stacked Kernel-Linear Hybrid Ensemble pipeline...")

    # 1. Load Data
    # load_dataset returns: X_train, y_train, X_val, y_val, X_test, test_ids, class_names
    X_train, y_train, X_val, y_val, X_test, test_ids, class_names = load_dataset(
        load_cached_data=True
    )

    print(
        f"Data Loaded: Train shape {X_train.shape}, Val shape {X_val.shape}, Test shape {X_test.shape}"
    )

    # 2. Initialize Ensemble
    ensemble = StackingEnsemble(random_state=RANDOM_SEED)

    # 3. Meta-Training (Level 1)
    # Generate OOF predictions on X_train to train the meta-learner
    print("\nStep 1: Generating OOF predictions for Meta-Learner training...")
    oof_preds = ensemble.generate_oof_predictions(X_train, y_train)

    print("Step 2: Training Meta-Learner...")
    ensemble.train_meta_learner(oof_preds, y_train)

    # 4. Validation (Hold-Out)
    # To evaluate on Val, we first need to train base models on X_train
    print("\nStep 3: Training Base Models on Training Set for Validation...")
    ensemble.train_full_base_models(X_train, y_train)

    print("Step 4: Evaluating on Hold-out Validation Set...")
    val_probs = ensemble.predict(X_val)

    # Calculate Metric
    # Ensure labels parameter is provided to handle potential missing classes in mini-batches (though unlikely here)
    labels = np.arange(len(class_names))
    val_metric = log_loss(y_val, val_probs, labels=labels)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    # We can use feature names if we reconstruct them, but indices are sufficient given the context
    # Feature order: Margin (64) -> Shape (64) -> Texture (64)
    feat_names = []
    for i in range(64):
        feat_names.append(f"margin_{i+1}")
    for i in range(64):
        feat_names.append(f"shape_{i+1}")
    for i in range(64):
        feat_names.append(f"texture_{i+1}")

    perform_failure_analysis(X_val, y_val, val_probs, feature_names=feat_names)

    # 6. Submission Logic
    if val_metric < METRIC_THRESHOLD:
        print(
            f"\nValidation metric meets threshold ({METRIC_THRESHOLD}). Proceeding to submission."
        )

        # Combine Train and Val for final retraining
        print("Step 5: Retraining Base Models on Full Dataset (Train + Val)...")
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        # Retrain base models
        ensemble.train_full_base_models(X_full, y_full)

        # Note: We keep the Meta-Learner trained on the OOF predictions from X_train.
        # This is a standard stacking practice (using OOF weights).

        print("Step 6: Generating Test Predictions...")
        test_probs = ensemble.predict(X_test)

        print("Step 7: Saving Submission...")
        save_submission(
            test_probs, test_ids, class_names, output_path="./submission/submission.csv"
        )

    else:
        print(
            f"\nValidation metric {val_metric} is higher than threshold {METRIC_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
