import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from library.configuration import Config
from library.utilities import seed_everything, compute_log_loss
from library.data_handling import get_stratified_folds, get_tfidf_features


def train_linear_expert(load_cached_data=True):
    """
    Trains the Surface Stylometric Expert (Logistic Regression on TF-IDF features).
    Implements 5-fold Cross-Validation to generate OOF predictions and averaged Test predictions.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed predictions from disk.

    Returns:
        tuple: (oof_preds, test_preds)
            oof_preds: np.array of shape (N_train, 3) containing validation probabilities.
            test_preds: np.array of shape (N_test, 3) containing averaged test probabilities.
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Cache file paths
    oof_cache_path = os.path.join(Config.WORKING_DIR, "oof_linear.npy")
    test_cache_path = os.path.join(Config.WORKING_DIR, "test_preds_linear.npy")

    # 1. Check Cache
    if (
        load_cached_data
        and os.path.exists(oof_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print("Loading cached Linear Expert predictions...")
        try:
            oof_preds = np.load(oof_cache_path)
            test_preds = np.load(test_cache_path)
            return oof_preds, test_preds
        except Exception as e:
            print(f"Failed to load cache: {e}. Proceeding to retrain...")

    print("Training Linear Expert (TF-IDF + Logistic Regression)...")

    # 2. Load Data
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # 3. Generate Features
    # This function handles its own caching for the sparse matrices
    print("Generating/Loading TF-IDF features...")
    X_train_sparse, X_test_sparse = get_tfidf_features(
        df_train, df_test, load_cached_data=load_cached_data
    )

    # 4. Prepare Folds
    df_train = get_stratified_folds(
        df_train, num_folds=Config.NUM_FOLDS, seed=Config.SEED
    )
    y = df_train["author"].map(Config.LABEL2ID).values

    # 5. Training Loop
    oof_preds = np.zeros((len(df_train), Config.NUM_CLASSES))
    test_preds_accumulator = np.zeros((len(df_test), Config.NUM_CLASSES))

    for fold in range(Config.NUM_FOLDS):
        # Identify indices
        train_idx = df_train[df_train["fold"] != fold].index
        val_idx = df_train[df_train["fold"] == fold].index

        # Slice sparse matrices
        X_tr_fold = X_train_sparse[train_idx]
        y_tr_fold = y[train_idx]
        X_val_fold = X_train_sparse[val_idx]
        y_val_fold = y[val_idx]

        # Initialize Model
        model = LogisticRegression(
            C=Config.LR_C,
            solver=Config.LR_SOLVER,
            max_iter=Config.LR_MAX_ITER,
            multi_class="multinomial",
            n_jobs=-1,
            random_state=Config.SEED,
        )

        # Train
        model.fit(X_tr_fold, y_tr_fold)

        # Predict Validation
        val_probs = model.predict_proba(X_val_fold)
        oof_preds[val_idx] = val_probs

        # Predict Test
        test_probs = model.predict_proba(X_test_sparse)
        test_preds_accumulator += test_probs

        # Evaluate
        fold_score = compute_log_loss(y_val_fold, val_probs)
        print(f"Fold {fold + 1} LogLoss: {fold_score}")

    # 6. Aggregate Test Predictions
    test_preds = test_preds_accumulator / Config.NUM_FOLDS

    # 7. Save to Cache
    try:
        np.save(oof_cache_path, oof_preds)
        np.save(test_cache_path, test_preds)
        print("Linear Expert predictions cached.")
    except Exception as e:
        print(f"Warning: Failed to save cache. {e}")

    return oof_preds, test_preds
