import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

from library.config import SUBMISSION_FILE, PROB_CLIP_EPS
from library.data_manager import load_combined_train_val, load_dataset
from library.models import get_linear_branch, get_generative_branch, get_kernel_branch


def train_and_predict_ensemble(load_cached_data=True):
    """
    Orchestrates the training of the ensemble and generates predictions.

    This function:
    1. Loads the augmented datasets (combining Train and Val for final training).
    2. Applies preprocessing (StandardScaler) where required.
    3. Trains three distinct model branches:
        - Discriminative Linear (LogisticRegressionCV)
        - Generative Linear (LDA)
        - Discriminative Kernel (Nystroem + LogisticRegressionCV)
    4. Ensembles predictions using Soft Voting.
    5. Saves the formatted submission file.

    Args:
        load_cached_data (bool): Whether to use cached feature files.
    """
    # 1. Load Data
    print("Loading combined training and validation data...")
    # We use the combined set for the final model to maximize signal
    X_train, y_train, ids_train = load_combined_train_val(
        load_cached_data=load_cached_data
    )

    print("Loading test data...")
    X_test, _, ids_test = load_dataset("test", load_cached_data=load_cached_data)

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # 2. Preprocessing
    # The Linear and Generative branches require explicit scaling.
    # The Kernel branch contains a StandardScaler within its pipeline.
    print("Preprocessing: Applying global scaling for Linear/Generative branches...")
    scaler = StandardScaler()

    # Fit on training data, transform both
    # Note: X_train is a DataFrame, output is a numpy array
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model Training

    # --- Branch 1: Discriminative Linear (Logistic Regression) ---
    print("Training Linear Branch (LogisticRegressionCV)...")
    clf_linear = get_linear_branch()
    clf_linear.fit(X_train_scaled, y_train)
    print("Linear Branch training complete.")

    # --- Branch 2: Generative Linear (LDA) ---
    print("Training Generative Branch (LDA)...")
    clf_gen = get_generative_branch()
    clf_gen.fit(X_train_scaled, y_train)
    print("Generative Branch training complete.")

    # --- Branch 3: Discriminative Kernel (Nystroem + LR) ---
    print("Training Kernel Branch (Nystroem Pipeline)...")
    # Pass unscaled data (DataFrame) as the pipeline handles scaling internally
    clf_kernel = get_kernel_branch()
    clf_kernel.fit(X_train, y_train)
    print("Kernel Branch training complete.")

    # 4. Inference
    print("Generating predictions on test set...")

    # Verify class alignment
    classes = clf_linear.classes_
    if not np.array_equal(clf_gen.classes_, classes):
        raise RuntimeError("Class mismatch between Linear and Generative models.")
    if not np.array_equal(clf_kernel.classes_, classes):
        raise RuntimeError("Class mismatch between Linear and Kernel models.")

    print(f"Number of classes: {len(classes)}")

    # Get probabilities from each branch
    # Linear and Gen use scaled data
    probs_linear = clf_linear.predict_proba(X_test_scaled)
    probs_gen = clf_gen.predict_proba(X_test_scaled)

    # Kernel uses raw data (pipeline scales)
    probs_kernel = clf_kernel.predict_proba(X_test)

    # 5. Ensemble (Soft Voting)
    print("Ensembling predictions via Soft Voting...")
    # Simple average of probabilities
    final_probs = (probs_linear + probs_gen + probs_kernel) / 3.0

    # 6. Post-processing
    # Clip probabilities to prevent infinite log loss
    # max(min(p, 1-eps), eps)
    final_probs = np.clip(final_probs, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)

    # 7. Save Submission
    print("Formatting and saving submission...")
    submission_df = pd.DataFrame(final_probs, columns=classes)
    submission_df.insert(0, "id", ids_test)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)

    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission successfully saved to {SUBMISSION_FILE}")
