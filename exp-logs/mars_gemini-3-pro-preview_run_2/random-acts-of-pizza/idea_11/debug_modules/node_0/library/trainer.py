import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from library.config import Config
from library.utils import set_seed, save_object, load_object
from library.data_loader import DataLoader
from library.features import get_feature_pipeline
from library.model import get_model, tune_hyperparameters


def run_cv_training():
    """
    Executes the 5-Fold Stratified Cross-Validation training loop.

    Process:
    1. Loads training data.
    2. Splits data into 5 stratified folds.
    3. For each fold:
       - Tunes hyperparameters using inner CV on the training split.
       - Fits the feature engineering pipeline and model on the training split.
       - Evaluates on the validation split.
       - Saves the fitted pipeline for inference.
    4. Reports mean and std ROC AUC.
    """
    set_seed(Config.SEED)

    print("Initializing Data Loader...")
    loader = DataLoader()

    # Load training data (merged with metadata)
    # The loader handles caching internally
    df_train = loader.load_merged_data(split="train")

    # Separate Features and Target
    # The feature pipeline handles column selection, so we pass the full DataFrame
    X = df_train.drop(columns=["requester_received_pizza"])
    y = df_train["requester_received_pizza"]

    # Initialize Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_scores = []

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*20} Fold {fold + 1}/{Config.N_FOLDS} {'='*20}")

        # Create Fold Splits
        # Using .iloc to slice DataFrames/Series based on integer indices
        X_fold_train = X.iloc[train_idx].reset_index(drop=True)
        y_fold_train = y.iloc[train_idx].reset_index(drop=True)
        X_fold_val = X.iloc[val_idx].reset_index(drop=True)
        y_fold_val = y.iloc[val_idx].reset_index(drop=True)

        print(f"Train shape: {X_fold_train.shape}, Val shape: {X_fold_val.shape}")

        # Step 1: Hyperparameter Tuning
        # We tune using the training portion of the current fold to prevent leakage.
        # This function performs an inner CV to find best C and class_weight.
        print("Step 1: Tuning hyperparameters...")
        best_params = tune_hyperparameters(X_fold_train, y_fold_train)
        print(f"Fold {fold + 1} Best Params: {best_params}")

        # Step 2: Model Construction
        # Build the pipeline with the best hyperparameters found.
        print("Step 2: Constructing and fitting final fold pipeline...")

        # Get fresh instances of the feature pipeline and model
        feature_pipeline = get_feature_pipeline()
        model = get_model(C=best_params["C"], class_weight=best_params["class_weight"])

        # Combine into a single executable pipeline
        # This ensures that raw data -> features -> prediction is encapsulated
        final_pipeline = Pipeline(
            [("preprocessor", feature_pipeline), ("classifier", model)]
        )

        # Step 3: Training
        # Fit on the full training set of this fold
        final_pipeline.fit(X_fold_train, y_fold_train)

        # Step 4: Evaluation
        print("Step 3: Evaluating on validation set...")
        # Predict probabilities for the positive class (1)
        y_pred_proba = final_pipeline.predict_proba(X_fold_val)[:, 1]

        # Calculate ROC AUC
        score = roc_auc_score(y_fold_val, y_pred_proba)
        fold_scores.append(score)

        # Print full precision score
        print(f"Fold {fold + 1} ROC AUC: {score}")

        # Step 5: Save Artifacts
        # Save the entire fitted pipeline (including feature transformers)
        model_filename = f"fold_{fold}_pipeline.joblib"
        model_path = os.path.join(Config.WORKING_DIR, model_filename)
        save_object(final_pipeline, model_path)
        print(f"Saved pipeline to {model_path}")

    # Final Summary
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    print(f"\n{'='*20} Cross-Validation Summary {'='*20}")
    print(f"Mean ROC AUC: {mean_score}")
    print(f"Std ROC AUC: {std_score}")
    print("=" * 60)

    return fold_scores


def generate_submission():
    """
    Generates predictions for the test set using the ensemble of 5 fold-models.
    Saves the result to the submission file defined in Config.
    """
    set_seed(Config.SEED)
    print("\nStarting Submission Generation...")

    # Load Test Data
    loader = DataLoader()
    df_test = loader.load_merged_data(split="test")
    X_test = df_test  # Pipeline handles column selection

    # Store predictions from each fold
    fold_preds = []

    for fold in range(Config.N_FOLDS):
        model_filename = f"fold_{fold}_pipeline.joblib"
        model_path = os.path.join(Config.WORKING_DIR, model_filename)

        print(f"Loading model from {model_path}...")
        try:
            pipeline = load_object(model_path)

            # Predict
            # predict_proba returns [prob_0, prob_1]
            preds = pipeline.predict_proba(X_test)[:, 1]
            fold_preds.append(preds)

        except Exception as e:
            print(f"Error loading/predicting with fold {fold}: {e}")
            raise e

    # Average predictions (Ensemble of Ensembles)
    # This reduces variance compared to using a single best model
    avg_preds = np.mean(fold_preds, axis=0)

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {"request_id": df_test["request_id"], "requester_received_pizza": avg_preds}
    )

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())
