import pandas as pd
import numpy as np
import gc
import os
from sklearn.metrics import accuracy_score
from library.config import DATA_PATHS, PIPELINE_PARAMS, TARGET_COL, ID_COL
from library.data_utils import (
    load_dataset,
    get_stratified_folds,
    create_augmented_train,
)
from library.model_utils import train_model, predict_proba, generate_submission, predict


def run_cross_validation(train_df, test_df, pseudo_df=None, n_folds=5, random_state=42):
    """
    Executes Stratified K-Fold Cross Validation.

    Args:
        train_df (pd.DataFrame): The original training data (including target).
        test_df (pd.DataFrame): The test data (features only).
        pseudo_df (pd.DataFrame, optional): Pseudo-labeled data to augment training folds.
        n_folds (int): Number of folds.
        random_state (int): Seed.

    Returns:
        tuple: (list of trained models, np.ndarray of averaged test probabilities)
    """
    # Identify feature columns
    feature_cols = [c for c in train_df.columns if c not in [TARGET_COL, ID_COL]]

    # Get folds based on ORIGINAL training data to preserve validation integrity
    folds = get_stratified_folds(train_df, n_folds=n_folds, random_state=random_state)

    models = []
    test_probs_sum = None
    val_scores = []

    print(f"Starting {n_folds}-Fold Cross-Validation...")

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

        # Split original data
        X_train = train_df.iloc[train_idx][feature_cols]
        y_train = train_df.iloc[train_idx][TARGET_COL]
        X_val = train_df.iloc[val_idx][feature_cols]
        y_val = train_df.iloc[val_idx][TARGET_COL]

        # Augment with pseudo-labels if provided
        if pseudo_df is not None:
            print(f"Augmenting fold with {len(pseudo_df)} pseudo-labeled samples.")
            X_pseudo = pseudo_df[feature_cols]
            y_pseudo = pseudo_df[TARGET_COL]

            X_train = pd.concat([X_train, X_pseudo], axis=0, ignore_index=True)
            y_train = pd.concat([y_train, y_pseudo], axis=0, ignore_index=True)

        # Train model
        model = train_model(X_train, y_train, X_val, y_val)
        models.append(model)

        # Predict on Test Set
        probs = predict_proba(model, test_df[feature_cols])

        if test_probs_sum is None:
            test_probs_sum = probs
        else:
            test_probs_sum += probs

        # Calculate and print validation accuracy with full precision
        val_preds = predict(model, X_val)
        acc = accuracy_score(y_val, val_preds)
        val_scores.append(acc)
        print(f"Fold {fold_idx + 1} Accuracy: {acc}")

        # Cleanup
        del X_train, y_train, X_val, y_val, val_preds, probs
        gc.collect()

    avg_test_probs = test_probs_sum / n_folds
    mean_val_score = np.mean(val_scores)
    print(f"\nAverage Validation Accuracy: {mean_val_score}")

    return models, avg_test_probs


def run_self_training(load_cached_data=True):
    """
    Orchestrates the two-stage self-training pipeline.
    """
    # 1. Load Data
    train_df, val_df, test_df = load_dataset(load_cached_data=load_cached_data)

    # Merge train and val to maximize data (Original Train Set)
    print("Merging Train and Validation sets for Cross-Validation...")
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # Clean up individual dfs to save memory
    del train_df, val_df
    gc.collect()

    # 2. Stage 1: Initial Bagging (Teacher)
    print("\n========================================")
    print("STAGE 1: Initial Bagging (Teacher)")
    print("========================================")

    models_s1, test_probs_s1 = run_cross_validation(
        full_train_df,
        test_df,
        pseudo_df=None,
        n_folds=PIPELINE_PARAMS["n_folds"],
        random_state=PIPELINE_PARAMS["random_state"],
    )

    # 3. Pseudo-Labeling
    print("\n========================================")
    print("PSEUDO-LABELING")
    print("========================================")

    # Use library function to generate augmented set
    aug_train_df = create_augmented_train(
        full_train_df,
        test_df,
        test_probs_s1,
        threshold=PIPELINE_PARAMS["pseudo_label_threshold"],
    )

    # Extract just the pseudo-labeled part to handle fold augmentation correctly
    n_original = len(full_train_df)
    if len(aug_train_df) > n_original:
        pseudo_df = aug_train_df.iloc[n_original:].copy()
        print(f"Extracted {len(pseudo_df)} pseudo-labeled samples.")
    else:
        pseudo_df = None
        print("No samples met the threshold. Skipping augmentation.")

    # Clean up Stage 1 models/probs
    del models_s1, test_probs_s1, aug_train_df
    gc.collect()

    # 4. Stage 2: Refinement (Student)
    print("\n========================================")
    print("STAGE 2: Refinement (Student)")
    print("========================================")

    models_s2, test_probs_s2 = run_cross_validation(
        full_train_df,
        test_df,
        pseudo_df=pseudo_df,
        n_folds=PIPELINE_PARAMS["n_folds"],
        random_state=PIPELINE_PARAMS["random_state"],
    )

    # 5. Submission
    print("\n========================================")
    print("GENERATING SUBMISSION")
    print("========================================")

    # Prepare features for submission generation
    feature_cols = [c for c in full_train_df.columns if c not in [TARGET_COL, ID_COL]]
    X_test = test_df[feature_cols]

    generate_submission(models_s2, X_test, DATA_PATHS["submission_output"])

    print("Pipeline completed successfully.")
