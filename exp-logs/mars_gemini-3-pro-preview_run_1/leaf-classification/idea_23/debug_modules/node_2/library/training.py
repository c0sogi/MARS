import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library import config
from library import data_processing
from library.model import TaxonomicDualCentroidOAS


def optimize_shrinkage(
    X, y, genus, classes, lambda_values=None, n_splits=5, seed=config.SEED
):
    """
    Executes a grid search for the optimal shrinkage factor lambda using Stratified K-Fold CV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target species labels.
        genus (np.ndarray): Genus labels corresponding to y.
        classes (np.ndarray): List of all unique class names for log_loss.
        lambda_values (np.ndarray, optional): Array of lambda values to test.
                                              Defaults to np.linspace(0, 0.5, 11).
        n_splits (int): Number of CV folds.
        seed (int): Random seed.

    Returns:
        float: The optimal lambda value minimizing log loss.
    """
    if lambda_values is None:
        lambda_values = np.linspace(0, 0.5, 11)

    print(f"Starting Grid Search for Lambda with {n_splits} folds...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    best_score = float("inf")
    best_lambda = lambda_values[0]

    # Pre-calculate splits to ensure consistency across lambda evaluations
    splits = list(skf.split(X, y))

    for l_val in lambda_values:
        fold_scores = []

        for train_idx, dev_idx in splits:
            X_train_fold, X_dev_fold = X[train_idx], X[dev_idx]
            y_train_fold, y_dev_fold = y[train_idx], y[dev_idx]
            genus_train_fold = genus[train_idx]

            # Instantiate and fit model
            model = TaxonomicDualCentroidOAS(lambda_reg=l_val)
            model.fit(X_train_fold, y_train_fold, genus_train_fold)

            # Predict
            preds = model.predict_proba(X_dev_fold)

            # Score
            score = log_loss(y_dev_fold, preds, labels=classes)
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        print(f"Lambda: {l_val:.4f} | CV Log Loss: {avg_score:.15f}")

        if avg_score < best_score:
            best_score = avg_score
            best_lambda = l_val

    print(f"Optimal Lambda found: {best_lambda:.4f} with CV Score: {best_score:.15f}")
    return best_lambda


def evaluate_model(model, X_val, y_val, classes):
    """
    Calculates and reports the final log loss on the validation set.

    Args:
        model: Trained TaxonomicDualCentroidOAS model.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation targets.
        classes (np.ndarray): Class labels.

    Returns:
        float: Multi-class log loss.
    """
    print("Evaluating model on validation set...")
    val_probs = model.predict_proba(X_val)
    val_loss = log_loss(y_val, val_probs, labels=classes)
    print(f"Validation Multi-class Log Loss: {val_loss:.15f}")
    return val_loss


def generate_submission(model, X_test, ids_test, classes, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: Trained model.
        X_test (np.ndarray): Test features.
        ids_test (np.ndarray): Test IDs.
        classes (np.ndarray): Class names (column headers).
        output_path (str): Path to save the submission CSV.
    """
    print("Generating test predictions...")
    test_probs = model.predict_proba(X_test)

    # Create DataFrame
    submission_df = pd.DataFrame(test_probs, columns=classes)
    submission_df.insert(0, "id", ids_test)

    print(f"Saving submission to {output_path}...")
    submission_df.to_csv(output_path, index=False)
    print("Submission saved successfully.")


def run_training(sample_size=None, n_splits=5):
    """
    Main pipeline execution function.

    Args:
        sample_size (int, optional): If provided, subsets the training data for debugging.
        n_splits (int): Number of cross-validation folds.
    """
    # 1. Load Data
    print("Loading data via data_processing module...")
    (
        X_train,
        y_train,
        genus_train,
        X_val,
        y_val,
        genus_val,
        X_test,
        ids_test,
        classes,
    ) = data_processing.process_data(load_cached_data=True)

    # Debugging: Subsample if requested
    if sample_size is not None and sample_size < len(X_train):
        print(f"Subsampling training data to {sample_size} samples for debugging.")
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        genus_train = genus_train[indices]

    # 2. Hyperparameter Optimization
    best_lambda = optimize_shrinkage(
        X_train,
        y_train,
        genus_train,
        classes,
        lambda_values=np.linspace(0, 0.5, 11),
        n_splits=n_splits,
    )

    # 3. Final Training
    print(
        f"Retraining final model with Lambda={best_lambda:.4f} on full training set..."
    )
    final_model = TaxonomicDualCentroidOAS(lambda_reg=best_lambda)
    final_model.fit(X_train, y_train, genus_train)

    # 4. Evaluation
    evaluate_model(final_model, X_val, y_val, classes)

    # 5. Submission
    generate_submission(final_model, X_test, ids_test, classes, config.SUBMISSION_PATH)
