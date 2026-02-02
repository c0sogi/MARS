import os
import numpy as np
import pandas as pd
from library import config
from library import utils
from library import preprocessing
from library import model


def run_solver(load_cached_data: bool = True):
    """
    Executes the Semi-Supervised Fixed-Mean Transduction workflow.

    1. Loads and preprocesses data (Yeo-Johnson + StandardScaler).
    2. Fits Class Means on Training Data (Fixed).
    3. Estimates Initial Covariance on Training Residuals.
    4. Generates Pseudo-labels for high-confidence Test samples.
    5. Refines Covariance using combined Training and Test Residuals (Transduction).
    6. Generates final predictions and submission file.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # 1. Setup
    utils.set_seed(config.RANDOM_SEED)
    print("Starting Solver: Semi-Supervised Fixed-Mean Transduction")

    # 2. Data Loading & Preprocessing
    # Returns float32 arrays and original labels/IDs
    X_train, y_train, X_val, y_val, X_test, test_ids = preprocessing.preprocess_data(
        load_cached_data=load_cached_data
    )

    print(f"Data Loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 3. Model Initialization
    clf = model.FixedMeanOASDiscriminant()

    # 4. Phase 1: Supervised Learning (Fixed Means)
    print("Phase 1: Fitting Class Means on Training Data...")
    clf.fit_means(X_train, y_train)

    print("Phase 1: Estimating Initial Covariance on Training Residuals...")
    residuals_train = clf.compute_residuals(X_train, y_train)
    clf.fit_covariance(residuals_train)

    # 5. Initial Validation
    print("Evaluating Baseline Model on Validation Set...")
    val_probs_init = clf.predict_proba(X_val)
    val_loss_init = utils.compute_log_loss(y_val, val_probs_init, labels=clf.classes_)
    print(f"Baseline Validation Log Loss: {val_loss_init}")

    # 6. Phase 2: Transduction (Covariance Refinement)
    print("Phase 2: Generating Initial Test Predictions for Pseudo-Labeling...")
    test_probs_init = clf.predict_proba(X_test)

    # Identify high confidence samples
    max_probs = np.max(test_probs_init, axis=1)
    high_conf_mask = max_probs > config.CONFIDENCE_THRESHOLD
    n_high_conf = np.sum(high_conf_mask)

    print(
        f"Found {n_high_conf} high-confidence test samples (Threshold > {config.CONFIDENCE_THRESHOLD})"
    )

    if n_high_conf > 0:
        # Extract pseudo-labels
        # argmax gives index into clf.classes_
        pseudo_label_indices = np.argmax(test_probs_init[high_conf_mask], axis=1)
        pseudo_labels = clf.classes_[pseudo_label_indices]

        # Get corresponding features
        X_test_conf = X_test[high_conf_mask]

        # Compute residuals for test samples using FIXED means
        # This is crucial: we do not update means, we only use the manifold structure
        print("Computing Test Residuals using Fixed Means...")
        residuals_test = clf.compute_residuals(X_test_conf, pseudo_labels)

        # Combine residuals
        print("Refining Covariance with Combined Residuals (Train + Pseudo-Test)...")
        residuals_combined = np.vstack([residuals_train, residuals_test])

        # Refit covariance
        clf.fit_covariance(residuals_combined)

        # Re-evaluate on Validation set
        # Note: Improvement on Val is not guaranteed as Transduction optimizes for Test distribution,
        # but it checks for stability.
        val_probs_refined = clf.predict_proba(X_val)
        val_loss_refined = utils.compute_log_loss(
            y_val, val_probs_refined, labels=clf.classes_
        )
        print(f"Refined Validation Log Loss: {val_loss_refined}")
    else:
        print("Skipping Covariance Refinement (insufficient high-confidence samples).")

    # 7. Final Inference
    print("Generating Final Test Predictions...")
    final_test_probs = clf.predict_proba(X_test)

    # 8. Submission Generation
    print("Creating Submission File...")

    # Create DataFrame
    # Columns must be: id, [class_names...]
    submission_df = pd.DataFrame(final_test_probs, columns=clf.classes_)
    submission_df.insert(0, config.ID_COL, test_ids)

    # Save
    submission_path = config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print("Solver execution complete.")
