import os
import joblib
from sklearn.metrics import roc_auc_score
from library import config
from library import utils
from library import model as lib_model
from library.data_processing import DataHandler


def train_model(X_train, y_train, X_val=None, y_val=None, max_samples=None):
    """
    Trains the LightGBM model.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training targets.
        X_val (array-like): Validation features.
        y_val (array-like): Validation targets.
        max_samples (int, optional): Limit training data size for debugging.

    Returns:
        lightgbm.LGBMClassifier: The trained model.
    """
    # Subset data for debugging if requested
    if max_samples is not None and max_samples < len(X_train):
        print(f"Subsetting training data to {max_samples} samples...")
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]

    # Instantiate model using library function to ensure consistent config
    model = lib_model.get_model()

    print("Fitting LightGBM model...")

    fit_params = {}
    if X_val is not None and y_val is not None:
        from lightgbm import early_stopping, log_evaluation

        fit_params = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": "auc",
            "callbacks": [
                early_stopping(stopping_rounds=50),
                log_evaluation(period=100),
            ],
        }

    model.fit(X_train, y_train, **fit_params)

    # Save the trained model
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

    return model


def evaluate_model(model, X, y):
    """
    Evaluates the model on a given dataset using ROC AUC.

    Args:
        model: Trained classifier.
        X (array-like): Features.
        y (array-like): True targets.

    Returns:
        float: Area Under the ROC Curve.
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    return auc


def predict_test(model, X_test, ids_test):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: Trained classifier.
        X_test (array-like): Test features.
        ids_test (array-like): Test IDs.
    """
    lib_model.generate_submission(model, X_test, ids_test)


def run(debug=False, load_cached_data=True):
    """
    Main pipeline execution function.

    Args:
        debug (bool): If True, runs on a subset of data.
        load_cached_data (bool): If True, attempts to load preprocessed data from cache.
    """
    # Ensure reproducibility
    utils.set_seed()

    # 1. Data Loading and Processing
    data_handler = DataHandler()
    X_train, y_train, X_val, y_val, X_test, ids_test = data_handler.get_processed_data(
        load_cached_data=load_cached_data
    )

    # Determine sample limit
    max_samples = config.DEBUG_SAMPLE_SIZE if debug else None

    # 2. Training
    model = train_model(X_train, y_train, X_val, y_val, max_samples=max_samples)

    # 3. Evaluation
    print("Calculating metrics...")
    # Evaluate on training set (subset if debug was used)
    train_subset_X = X_train[:max_samples] if max_samples else X_train
    train_subset_y = y_train[:max_samples] if max_samples else y_train

    train_auc = evaluate_model(model, train_subset_X, train_subset_y)
    val_auc = evaluate_model(model, X_val, y_val)

    print(f"Training AUC: {train_auc}")
    print(f"Validation AUC: {val_auc}")

    # 4. Prediction
    predict_test(model, X_test, ids_test)
