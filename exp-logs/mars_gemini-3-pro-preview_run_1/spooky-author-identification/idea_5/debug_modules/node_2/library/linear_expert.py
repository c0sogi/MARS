import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from library.utils import seed_everything, calculate_log_loss, ensure_directory
from library.feature_engineering import get_tfidf_vectorizer
from library.data_loader import create_stratified_folds, LABEL_MAP

# Define working directories
WORKING_DIR = "./working/idea_5/"
MODEL_DIR = os.path.join(WORKING_DIR, "linear_models")


def train_predict_linear_fold(
    X_train_text, y_train, X_val_text, X_test_text, seed=42, C=1.0
):
    """
    Trains a Logistic Regression model on TF-IDF features for a single fold.

    Args:
        X_train_text (pd.Series): Training text.
        y_train (np.array): Training labels.
        X_val_text (pd.Series): Validation text.
        X_test_text (pd.Series): Test text.
        seed (int): Random seed.
        C (float): Inverse of regularization strength.

    Returns:
        tuple: (val_probs, test_probs, model)
    """
    # Initialize Vectorizer
    # We fit strictly on the training data of this fold to avoid leakage
    vectorizer = get_tfidf_vectorizer()

    # Fit and Transform
    vectorizer.fit(X_train_text)
    X_train_tfidf = vectorizer.transform(X_train_text)
    X_val_tfidf = vectorizer.transform(X_val_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    # Initialize and Train Model
    # 'sag' or 'saga' are faster for large datasets, but 'lbfgs' is default and robust.
    # Given "trains in seconds", default is likely fine, but 'sag' is often preferred for text.
    clf = LogisticRegression(
        C=C,
        solver="sag",
        multi_class="multinomial",
        random_state=seed,
        n_jobs=-1,
        max_iter=1000,  # Ensure convergence
    )
    clf.fit(X_train_tfidf, y_train)

    # Predict Probabilities
    val_probs = clf.predict_proba(X_val_tfidf)
    test_probs = clf.predict_proba(X_test_tfidf)

    return val_probs, test_probs, clf


def run_linear_expert(n_folds=5, seed=42, debug=False, load_cached_data=True, C=1.0):
    """
    Runs the 5-Fold Cross-Validation for the Linear Expert.
    Generates OOF predictions and averaged Test predictions.

    Args:
        n_folds (int): Number of folds.
        seed (int): Random seed.
        debug (bool): Whether to run in debug mode (subset of data).
        load_cached_data (bool): Whether to load predictions from cache.
        C (float): Logistic Regression regularization parameter.

    Returns:
        tuple: (oof_preds, test_preds)
    """
    seed_everything(seed)
    ensure_directory(WORKING_DIR)
    ensure_directory(MODEL_DIR)

    # Cache paths
    oof_path = os.path.join(WORKING_DIR, "oof_linear.npy")
    test_pred_path = os.path.join(WORKING_DIR, "test_linear.npy")

    # 1. Check Cache
    if load_cached_data and os.path.exists(oof_path) and os.path.exists(test_pred_path):
        print(f"Loading cached Linear Expert predictions from {WORKING_DIR}...")
        try:
            oof_preds = np.load(oof_path)
            test_preds = np.load(test_pred_path)
            return oof_preds, test_preds
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Running Linear Expert (TF-IDF + Logistic Regression)...")

    # 2. Load Data
    # Load Folds
    df_folds = create_stratified_folds(
        data_path="./metadata/train.csv",
        n_folds=n_folds,
        seed=seed,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # Load Test Data
    df_test = pd.read_csv("./metadata/test.csv")
    if debug:
        # Sample test set for consistency if debugging
        df_test = df_test.head(1000)
        print(f"Debug mode: Sampled {len(df_test)} test rows.")

    # Prepare containers
    # OOF preds: (n_train, 3)
    oof_preds = np.zeros((len(df_folds), 3))
    # Test preds accumulator: (n_test, 3)
    test_preds_accum = np.zeros((len(df_test), 3))

    # Map labels to integers
    # LABEL_MAP = {"EAP": 0, "HPL": 1, "MWS": 2}
    df_folds["label_idx"] = df_folds["author"].map(LABEL_MAP)

    scores = []

    # 3. Cross-Validation Loop
    for fold in range(n_folds):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # Split Data
        train_idx = df_folds["fold"] != fold
        val_idx = df_folds["fold"] == fold

        df_train_fold = df_folds[train_idx]
        df_val_fold = df_folds[val_idx]

        # Extract text and labels
        # Fill NaNs just in case
        X_train_text = df_train_fold["text"].fillna("").astype(str)
        y_train = df_train_fold["label_idx"].values

        X_val_text = df_val_fold["text"].fillna("").astype(str)
        y_val = df_val_fold["label_idx"].values

        X_test_text = df_test["text"].fillna("").astype(str)

        # Train and Predict
        val_probs, test_probs, model = train_predict_linear_fold(
            X_train_text, y_train, X_val_text, X_test_text, seed=seed, C=C
        )

        # Store OOF
        # We use the index from the dataframe to map back to the original array
        # Assuming df_folds index aligns with 0..N-1 or we use iloc logic on the array
        # Since create_stratified_folds preserves original order and index, we can use boolean indexing
        # on the oof_preds array directly if we iterate carefully, or use indices.
        # Safest is to use the integer indices of the validation set.
        val_indices = df_folds.index[val_idx]
        oof_preds[val_indices] = val_probs

        # Accumulate Test Preds
        test_preds_accum += test_probs

        # Calculate Score
        fold_loss = calculate_log_loss(y_val, val_probs)
        scores.append(fold_loss)
        print(f"Fold {fold + 1} Log Loss: {fold_loss}")

        # Save Model
        model_path = os.path.join(MODEL_DIR, f"linear_model_fold_{fold}.joblib")
        joblib.dump(model, model_path)

    # 4. Aggregate
    test_preds = test_preds_accum / n_folds

    overall_loss = calculate_log_loss(df_folds["label_idx"].values, oof_preds)
    print(f"\nOverall CV Log Loss: {overall_loss}")
    print(f"Average Fold Log Loss: {np.mean(scores)}")

    # 5. Save Predictions
    try:
        np.save(oof_path, oof_preds)
        np.save(test_pred_path, test_preds)
        print(f"Saved OOF predictions to {oof_path}")
        print(f"Saved Test predictions to {test_pred_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")

    return oof_preds, test_preds
