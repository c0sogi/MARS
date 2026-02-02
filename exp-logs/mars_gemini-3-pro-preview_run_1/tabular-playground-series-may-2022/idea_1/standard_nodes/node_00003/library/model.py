import os
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from library import config
from library import utils


def get_model(**kwargs):
    """
    Instantiates and returns a LightGBM model with the specified parameters.
    Cite solution_lesson_node_00001: Switching to non-linear algorithm.

    Returns:
        lightgbm.LGBMClassifier: The configured model instance.
    """
    params = config.LGBM_PARAMS.copy()
    params.update(kwargs)
    model = LGBMClassifier(**params)
    return model


def train_model(X_train, y_train, X_val, y_val, max_samples=None):
    """
    Trains the LightGBM model, evaluates it on the validation set,
    prints metrics, and saves the model.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training targets.
        X_val (array-like): Validation features.
        y_val (array-like): Validation targets.
        max_samples (int, optional): If provided, limits the training data size for debugging.

    Returns:
        lightgbm.LGBMClassifier: The trained model.
    """
    # Handle debugging/subsetting if requested
    if max_samples is not None and max_samples < len(X_train):
        print(f"Subsetting training data to {max_samples} samples for debugging...")
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]

    # Instantiate model
    model = get_model()

    print("Fitting LightGBM model...")

    from lightgbm import early_stopping, log_evaluation

    callbacks = [
        early_stopping(stopping_rounds=50, verbose=True),
        log_evaluation(period=100),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric="auc",
        callbacks=callbacks,
    )

    # Evaluate on Training set
    print("Calculating training metrics...")
    y_pred_train = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_pred_train)
    print(f"Training AUC: {train_auc}")

    # Evaluate on Validation set
    print("Calculating validation metrics...")
    y_pred_val = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_pred_val)
    print(f"Validation AUC: {val_auc}")

    # Save the model
    # Ensure directory exists (though config usually points to working/idea_1 which exists)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

    return model


def generate_submission(model, X_test, ids_test):
    """
    Generates predictions for the test set and saves them to the submission file.

    Args:
        model (sklearn.linear_model.LogisticRegression): The trained model.
        X_test (array-like): Test features.
        ids_test (array-like): Test IDs.
    """
    print("Generating predictions for test set...")
    # Predict probabilities for the positive class (state 1)
    predictions = model.predict_proba(X_test)[:, 1]

    # Save submission using the utility function
    utils.save_submission(ids_test, predictions)
