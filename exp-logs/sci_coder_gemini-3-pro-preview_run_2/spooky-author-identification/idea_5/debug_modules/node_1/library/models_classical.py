import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from library.config import Config
from library.utils import save_numpy, load_numpy, save_model, load_model


def run_classical_models(
    df_train,
    df_test,
    train_tfidf,
    test_tfidf,
    train_svd,
    test_svd,
    load_cached_data=True,
):
    """
    Trains and generates predictions for classical models:
    1. Logistic Regression (Sparse TF-IDF)
    2. Multinomial Naive Bayes (Sparse TF-IDF)
    3. XGBoost (Dense SVD)

    Args:
        df_train (pd.DataFrame): Training metadata with 'fold' and 'author' columns.
        df_test (pd.DataFrame): Test metadata.
        train_tfidf (scipy.sparse.csr_matrix): TF-IDF features for training set.
        test_tfidf (scipy.sparse.csr_matrix): TF-IDF features for test set.
        train_svd (np.ndarray): SVD features for training set.
        test_svd (np.ndarray): SVD features for test set.
        load_cached_data (bool): Whether to load predictions from disk if available.

    Returns:
        tuple: (oof_preds, test_preds)
            oof_preds (dict): Dictionary containing OOF predictions for 'lr', 'nb', 'xgb'.
            test_preds (dict): Dictionary containing Test predictions for 'lr', 'nb', 'xgb'.
    """

    # Define cache filenames
    cache_files = {
        "oof_lr": "oof_lr.npy",
        "test_lr": "pred_test_lr.npy",
        "oof_nb": "oof_nb.npy",
        "test_nb": "pred_test_nb.npy",
        "oof_xgb": "oof_xgb.npy",
        "test_xgb": "pred_test_xgb.npy",
    }

    # Check if all cache files exist
    all_cached = all(
        os.path.exists(os.path.join(Config.WORKING_DIR, f))
        for f in cache_files.values()
    )

    if load_cached_data and all_cached:
        print("Loading classical model predictions from cache...")
        oof_preds = {
            "lr": load_numpy(cache_files["oof_lr"]),
            "nb": load_numpy(cache_files["oof_nb"]),
            "xgb": load_numpy(cache_files["oof_xgb"]),
        }
        test_preds = {
            "lr": load_numpy(cache_files["test_lr"]),
            "nb": load_numpy(cache_files["test_nb"]),
            "xgb": load_numpy(cache_files["test_xgb"]),
        }
        return oof_preds, test_preds

    print("Training classical models...")

    # Initialize containers
    n_train = len(df_train)
    n_test = len(df_test)
    num_classes = 3

    oof_preds = {
        "lr": np.zeros((n_train, num_classes)),
        "nb": np.zeros((n_train, num_classes)),
        "xgb": np.zeros((n_train, num_classes)),
    }

    test_preds = {
        "lr": np.zeros((n_test, num_classes)),
        "nb": np.zeros((n_test, num_classes)),
        "xgb": np.zeros((n_test, num_classes)),
    }

    # Map labels to integers
    label_map = {"EAP": 0, "HPL": 1, "MWS": 2}
    y = df_train["author"].map(label_map).values

    # Iterate over folds
    for fold in range(Config.NUM_FOLDS):
        print(f"\n--- Fold {fold + 1}/{Config.NUM_FOLDS} ---")

        # Split indices
        train_idx = df_train[df_train["fold"] != fold].index
        val_idx = df_train[df_train["fold"] == fold].index

        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # ==========================================
        # 1. Logistic Regression (TF-IDF)
        # ==========================================
        print("  Training Logistic Regression...")
        X_train_tfidf_fold = train_tfidf[train_idx]
        X_val_tfidf_fold = train_tfidf[val_idx]

        lr_model = LogisticRegression(
            C=1.0,
            solver="saga",
            multi_class="multinomial",
            max_iter=1000,
            random_state=Config.SEED,
            n_jobs=-1,
        )
        lr_model.fit(X_train_tfidf_fold, y_train_fold)

        # Predict
        p_val_lr = lr_model.predict_proba(X_val_tfidf_fold)
        p_test_lr = lr_model.predict_proba(test_tfidf)

        # Store
        oof_preds["lr"][val_idx] = p_val_lr
        test_preds["lr"] += p_test_lr / Config.NUM_FOLDS

        score_lr = log_loss(y_val_fold, p_val_lr)
        print(f"    LR Log Loss: {score_lr}")

        save_model(lr_model, f"model_lr_fold_{fold}.joblib", model_type="sklearn")

        # ==========================================
        # 2. Naive Bayes (TF-IDF)
        # ==========================================
        print("  Training Naive Bayes...")
        nb_model = MultinomialNB(alpha=0.02)
        nb_model.fit(X_train_tfidf_fold, y_train_fold)

        # Predict
        p_val_nb = nb_model.predict_proba(X_val_tfidf_fold)
        p_test_nb = nb_model.predict_proba(test_tfidf)

        # Store
        oof_preds["nb"][val_idx] = p_val_nb
        test_preds["nb"] += p_test_nb / Config.NUM_FOLDS

        score_nb = log_loss(y_val_fold, p_val_nb)
        print(f"    NB Log Loss: {score_nb}")

        save_model(nb_model, f"model_nb_fold_{fold}.joblib", model_type="sklearn")

        # ==========================================
        # 3. XGBoost (SVD)
        # ==========================================
        print("  Training XGBoost...")
        X_train_svd_fold = train_svd[train_idx]
        X_val_svd_fold = train_svd[val_idx]

        # Configure for GPU if available
        tree_method = "hist"
        device = "cuda" if Config.DEVICE.type == "cuda" else "cpu"

        xgb_model = XGBClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.6,
            objective="multi:softprob",
            num_class=3,
            random_state=Config.SEED,
            tree_method=tree_method,
            device=device,
            enable_categorical=False,
            n_jobs=-1 if device == "cpu" else 1,
            verbosity=0,
        )

        xgb_model.fit(
            X_train_svd_fold,
            y_train_fold,
            eval_set=[(X_val_svd_fold, y_val_fold)],
            early_stopping_rounds=50,
            verbose=False,
        )

        # Predict
        p_val_xgb = xgb_model.predict_proba(X_val_svd_fold)
        p_test_xgb = xgb_model.predict_proba(test_svd)

        # Store
        oof_preds["xgb"][val_idx] = p_val_xgb
        test_preds["xgb"] += p_test_xgb / Config.NUM_FOLDS

        score_xgb = log_loss(y_val_fold, p_val_xgb)
        print(f"    XGB Log Loss: {score_xgb}")

        # Save XGBoost model (using sklearn wrapper save method via joblib or internal)
        save_model(xgb_model, f"model_xgb_fold_{fold}.joblib", model_type="sklearn")

    # Save all predictions to cache
    print("\nSaving classical model predictions to cache...")
    save_numpy(oof_preds["lr"], cache_files["oof_lr"])
    save_numpy(test_preds["lr"], cache_files["test_lr"])

    save_numpy(oof_preds["nb"], cache_files["oof_nb"])
    save_numpy(test_preds["nb"], cache_files["test_nb"])

    save_numpy(oof_preds["xgb"], cache_files["oof_xgb"])
    save_numpy(test_preds["xgb"], cache_files["test_xgb"])

    return oof_preds, test_preds
