import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer

# Import from provided libraries
from library.config import (
    RANDOM_SEED,
    SUBMISSION_OUTPUT,
    N_JOBS,
    ORIGINAL_FEATURE_COLS,
    MORPHOLOGICAL_COLS,
    ID_COL,
    SUBMISSION_THRESHOLD,
)
from library.data_loader import LeafDataManager
from library.models import ExpertFactory
from library.ensemble import GreedyEnsembleSelector


# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_per_sample_log_loss(y_true, y_pred, classes):
    """
    Calculate log loss for each sample.
    y_true: array of shape (n_samples,) with class labels
    y_pred: array of shape (n_samples, n_classes) with probabilities
    classes: array of class labels corresponding to y_pred columns
    """
    # Create a mapping from class label to column index
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Get the predicted probability for the true class for each sample
    true_class_probs = []
    for true_label, pred_row in zip(y_true, y_pred):
        if true_label in class_to_idx:
            idx = class_to_idx[true_label]
            prob = pred_row[idx]
        else:
            # Should not happen if classes are correct, but handle safely
            prob = 1e-15

        # Clip to avoid log(0)
        prob = max(min(prob, 1 - 1e-15), 1e-15)
        true_class_probs.append(prob)

    true_class_probs = np.array(true_class_probs)
    # Log loss is negative log probability
    losses = -np.log(true_class_probs)
    return losses


def perform_failure_analysis(X_val_original, losses, feature_names):
    """
    Correlate per-sample loss with features to find sources of error.
    """
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    correlations = []

    # Ensure X_val_original is a numpy array
    X_matrix = np.array(X_val_original)

    # Calculate correlation for each feature
    for i, feature_name in enumerate(feature_names):
        if i >= X_matrix.shape[1]:
            break

        feat_values = X_matrix[:, i]
        # Handle constant features
        if np.std(feat_values) == 0:
            corr = 0
        else:
            corr = np.corrcoef(feat_values, losses)[0, 1]

        if np.isnan(corr):
            corr = 0
        correlations.append((feature_name, corr))

    # Sort by absolute correlation (magnitude of impact)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features associated with high error:")
    for name, corr in correlations[:5]:
        print(f"  - {name}: {corr:.4f}")


def main():
    set_seed(RANDOM_SEED)
    print("Starting MAGDE Pipeline...")

    # -------------------------------------------------------------------------
    # 1. Data Loading & Phase 1 Split
    # -------------------------------------------------------------------------
    print("Loading data and generating splits...")
    data_manager = LeafDataManager()

    # Get Phase 1 splits (Train/Val/Test)
    # load_cached_data=True to use existing parquet files if available
    X_train, y_train, X_val, y_val, X_test_phase1, test_ids_phase1, classes = (
        data_manager.get_splits(load_cached_data=True)
    )

    print(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")
    print(f"Classes: {len(classes)}")

    # -------------------------------------------------------------------------
    # 2. Expert Initialization
    # -------------------------------------------------------------------------
    print("Initializing Experts...")
    # Cite Lesson 70 (Gaussianization) and Lesson 73 (float64 precision)
    # We test variations of LDA solvers and tolerances on the Combined view (Cite Lesson 63)
    experts = {
        "Expert_C_Power_LSQR_Base": {
            "model": ExpertFactory.create_pipeline(
                preprocessor_type="power", model_type="lda", solver="lsqr"
            ),
            "view": "combined",
        },
        "Expert_C_Power_Eigen_Tight": {
            "model": ExpertFactory.create_pipeline(
                preprocessor_type="power",
                model_type="lda",
                solver="eigen",
                tol=1e-15,  # Tighter tolerance for float64
            ),
            "view": "combined",
        },
        "Expert_C_Quantile_LSQR": {
            "model": ExpertFactory.create_pipeline(
                preprocessor_type="quantile", model_type="lda", solver="lsqr"
            ),
            "view": "combined",
        },
        "Expert_D_Backup": {
            "model": ExpertFactory.create_pipeline(
                preprocessor_type="power", model_type="logreg"
            ),
            "view": "original",
        },
    }

    # -------------------------------------------------------------------------
    # 3. Phase 1: Training & Selection
    # -------------------------------------------------------------------------
    print("Phase 1: Training Experts on Training Split...")
    val_preds = {}

    for name, expert_info in experts.items():
        model = expert_info["model"]
        view_name = expert_info["view"]

        # Get appropriate view
        X_tr_view = X_train[view_name]
        X_val_view = X_val[view_name]

        print(f"  Training {name} on view '{view_name}' (shape: {X_tr_view.shape})...")
        model.fit(X_tr_view, y_train)

        # Predict on validation
        preds = model.predict_proba(X_val_view)
        val_preds[name] = preds

    print("Phase 1: Running Greedy Ensemble Selection...")
    selector = GreedyEnsembleSelector()
    selector.fit(val_preds, y_val, classes=classes)

    # Calculate Final Validation Metric (on hold-out set)
    # Using the selector to combine predictions
    final_val_preds = selector.predict(val_preds)
    final_val_metric = log_loss(y_val, final_val_preds, labels=classes)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    # Calculate per-sample loss
    losses = calculate_per_sample_log_loss(y_val, final_val_preds, classes)

    # Correlate with Original Features (most interpretable)
    # We use X_val['original'] which is transformed.
    # For analysis, transformed features are fine as they represent what the model sees.
    perform_failure_analysis(X_val["original"], losses, ORIGINAL_FEATURE_COLS)

    # -------------------------------------------------------------------------
    # 5. Phase 2: Final Retraining & Submission
    # -------------------------------------------------------------------------
    # Check submission condition
    # Using the strict threshold from the prompt/previous best

    if final_val_metric < SUBMISSION_THRESHOLD:
        print("\nPhase 2: Retraining Selected Experts on Full Data...")

        # Load Full Data
        X_full, y_full, X_test_full, test_ids, classes_full = (
            data_manager.get_full_data(load_cached_data=True)
        )

        test_preds_dict = {}

        # Iterate over experts and retrain only if they have weight > 0
        for name, expert_info in experts.items():
            weight = selector.weights.get(name, 0)
            if weight > 0:
                print(f"  Retraining {name} (Weight: {weight})...")
                model = expert_info["model"]  # Re-use the object (will be refitted)
                view_name = expert_info["view"]

                # Get full data view
                X_full_view = X_full[view_name]
                X_test_view = X_test_full[view_name]

                # Fit on full data
                model.fit(X_full_view, y_full)

                # Predict on test
                preds = model.predict_proba(X_test_view)
                test_preds_dict[name] = preds

        # Combine Test Predictions
        print("Generating Final Test Predictions...")
        final_test_preds = selector.predict(test_preds_dict)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(final_test_preds, columns=classes_full)
        submission_df.insert(0, ID_COL, test_ids)

        # Save
        print(f"Saving submission to {SUBMISSION_OUTPUT}...")
        submission_df.to_csv(SUBMISSION_OUTPUT, index=False)
        print("Submission saved successfully.")

    else:
        print(
            f"Validation metric {final_val_metric} is not lower than threshold {SUBMISSION_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
