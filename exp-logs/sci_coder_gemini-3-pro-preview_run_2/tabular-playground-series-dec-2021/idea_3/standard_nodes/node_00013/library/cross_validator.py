import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from library.config import SEED, N_FOLDS, TARGET_COL, ID_COL


def run_stratified_kfold(
    model_cls, model_params, df_train, df_test, n_folds=N_FOLDS, verbose=True
):
    """
    Executes Stratified K-Fold Cross-Validation for a given model class and parameters.

    Args:
        model_cls: The class of the model wrapper (e.g., LGBMWrapper, XGBWrapper).
        model_params (dict): Hyperparameters for the model.
        df_train (pd.DataFrame): The training dataset including the target column.
        df_test (pd.DataFrame): The test dataset.
        n_folds (int): Number of cross-validation folds.
        verbose (bool): If True, prints progress and metrics.

    Returns:
        tuple: (oof_preds, test_preds, classes)
            oof_preds (np.ndarray): Out-Of-Fold probability predictions for the training set.
            test_preds (np.ndarray): Averaged probability predictions for the test set.
            classes (np.ndarray): The unique class labels corresponding to the probability columns.
    """

    # --- Data Preparation ---
    # Separate features (X) and target (y) from the training dataframe
    # Drop ID_COL if present, as it's not a feature
    X = df_train.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    y = df_train[TARGET_COL]

    # Prepare test features
    X_test = df_test.drop(columns=[ID_COL, TARGET_COL], errors="ignore")

    # Determine classes and shapes
    # We assume the model wrapper uses LabelEncoder internally on sorted unique values of y
    classes = np.unique(y)
    n_classes = len(classes)
    n_train_samples = len(X)
    n_test_samples = len(X_test)

    # Initialize arrays to store predictions
    # oof_preds stores the probability vector for each training sample when it was in the validation fold
    oof_preds = np.zeros((n_train_samples, n_classes))
    # test_preds accumulates the probability vectors from each fold for the test set
    test_preds = np.zeros((n_test_samples, n_classes))

    # --- Stratified K-Fold Loop ---
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    if verbose:
        print(
            f"Starting Stratified K-Fold CV (K={n_folds}) for {model_cls.__name__}..."
        )
        print(f"Training Data Shape: {X.shape}")
        print(f"Test Data Shape: {X_test.shape}")
        print(f"Number of Classes: {n_classes}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        if verbose:
            print(f"\nProcessing Fold {fold + 1}/{n_folds}...")

        # Split data for this fold
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Instantiate the model
        model = model_cls(model_params)

        # Train the model
        # The wrapper is responsible for internal validation and early stopping
        model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

        # Generate predictions (probabilities)
        val_probs = model.predict_proba(X_val_fold)
        test_probs = model.predict_proba(X_test)

        # Store OOF predictions
        oof_preds[val_idx] = val_probs

        # Accumulate test predictions
        test_preds += test_probs

    # --- Aggregation and Evaluation ---

    # Average the test predictions across all folds
    test_preds /= n_folds

    # Calculate overall OOF metrics
    # Convert probability vectors to class labels for accuracy calculation
    # argmax returns the index of the max probability, which corresponds to the index in 'classes'
    oof_pred_indices = np.argmax(oof_preds, axis=1)
    oof_pred_labels = classes[oof_pred_indices]

    overall_acc = accuracy_score(y, oof_pred_labels)
    overall_logloss = log_loss(y, oof_preds)

    if verbose:
        print("\n" + "=" * 30)
        print(f"Cross-Validation Complete: {model_cls.__name__}")
        print("=" * 30)
        print(f"Overall OOF Accuracy: {overall_acc}")
        print(f"Overall OOF LogLoss: {overall_logloss}")

    return oof_preds, test_preds, classes
