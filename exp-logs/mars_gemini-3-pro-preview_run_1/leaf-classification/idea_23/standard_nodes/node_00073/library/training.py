import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library import config
from library import data_processing
from library.model import OASDiscriminant


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

    # 2. Final Training
    print("Training OASDiscriminant model on full training set...")
    final_model = OASDiscriminant()
    final_model.fit(X_train, y_train)

    # 3. Evaluation
    evaluate_model(final_model, X_val, y_val, classes)

    # 4. Submission
    generate_submission(final_model, X_test, ids_test, classes, config.SUBMISSION_PATH)
