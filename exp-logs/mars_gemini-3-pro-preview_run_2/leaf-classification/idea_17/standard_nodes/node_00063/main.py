import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.metrics import log_loss

# Import from provided libraries
from library.utils import set_seed, clip_and_score
from library.data_loader import load_dataset
from library.model_factory import get_linear_lda, get_discriminative_lr
from library.ensemble_selection import GreedySelector

# Constants
THRESHOLD = 3.3768e-06
SUBMISSION_DIR = "./submission"


def run():
    # 1. Setup
    set_seed(42)
    print("Starting execution...")

    # 2. Load Data
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_dataset(
        load_cached_data=True
    )

    # 3. Phase 1: Train Experts & Selection
    print("Initializing experts...")
    # Removed Kernel_LDA (Cite 00061)
    experts = {
        "Linear_LDA": get_linear_lda(random_state=42),
        "Linear_LR": get_discriminative_lr(random_state=42),
    }

    val_predictions = {}

    print("Training experts on training set...")
    for name, model in experts.items():
        print(f"  Fitting {name}...")
        model.fit(X_train, y_train)
        # Predict on validation set
        preds = model.predict_proba(X_val)
        val_predictions[name] = preds

        # Log individual performance
        loss = clip_and_score(y_val, preds)
        print(f"  {name} Val LogLoss: {loss:.6f}")

    print("Running Greedy Ensemble Selection...")
    selector = GreedySelector(iterations=100, random_state=42)
    selector.fit(val_predictions, y_val)

    weights = selector.get_weights()
    best_score = selector.best_score_

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {best_score}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Reconstruct ensemble predictions for validation set
    ensemble_preds = selector.predict(val_predictions)

    # Calculate per-sample log loss (Error Magnitude)
    # Rescale rows to sum to 1
    row_sums = ensemble_preds.sum(axis=1)
    row_sums[row_sums == 0] = 1
    ensemble_preds_norm = ensemble_preds / row_sums[:, np.newaxis]

    # Clip probabilities
    eps = 1e-15
    ensemble_preds_clipped = np.clip(ensemble_preds_norm, eps, 1 - eps)

    # Extract probability assigned to the true class
    # y_val contains integer class indices
    true_class_probs = ensemble_preds_clipped[np.arange(len(y_val)), y_val]
    sample_losses = -np.log(true_class_probs)

    # Calculate correlation between feature values and error magnitude
    correlations = []
    n_features = X_val.shape[1]

    # Check for constant features to avoid warnings
    for i in range(n_features):
        feature_vals = X_val[:, i]
        if np.std(feature_vals) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(feature_vals, sample_losses)
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for idx, corr in correlations[:5]:
        print(f"  Feature Index {idx}: Correlation = {corr:.4f}")

    # 5. Submission Logic (Conditional)
    if best_score < THRESHOLD:
        print(
            f"\nValidation score ({best_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Phase 2: Retrain on Full Data
        print("Retraining selected experts on full data (Train + Val)...")
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        final_models = {}
        for name, w in weights.items():
            if w > 0:
                print(f"  Retraining {name} (Weight: {w:.2f})...")
                # Clone to ensure a fresh model
                model = clone(experts[name])
                model.fit(X_full, y_full)
                final_models[name] = model

        # Phase 3: Inference on Test Set
        print("Predicting on Test set...")
        test_probs_sum = np.zeros((len(X_test), len(classes)))

        for name, model in final_models.items():
            w = weights[name]
            probs = model.predict_proba(X_test)
            test_probs_sum += w * probs

        # Normalize final probabilities
        row_sums_test = test_probs_sum.sum(axis=1)
        row_sums_test[row_sums_test == 0] = 1.0
        final_test_probs = test_probs_sum / row_sums_test[:, np.newaxis]

        # Save Submission
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

        df_sub = pd.DataFrame(final_test_probs, columns=classes)
        df_sub.insert(0, "id", test_ids)
        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation score ({best_score}) does not meet threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    run()
