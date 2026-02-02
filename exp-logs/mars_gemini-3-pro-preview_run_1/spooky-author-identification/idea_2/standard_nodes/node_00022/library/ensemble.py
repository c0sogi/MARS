import numpy as np
import xgboost as xgb
from library.utils import compute_log_loss
from library.config import Config
from library.data_loader import load_data


def get_meta_features(df):
    """
    Extracts simple meta-features for stacking.
    Cite solution_lesson_node_00008: Conditional Ensembling via Meta-Feature Stacking
    """
    texts = df["text"].fillna("").astype(str)
    meta = pd.DataFrame()
    meta["char_len"] = texts.apply(len)
    meta["word_count"] = texts.apply(lambda x: len(x.split()))
    # Normalize roughly to help tree convergence
    meta["char_len"] = meta["char_len"] / 150.0
    meta["word_count"] = meta["word_count"] / 25.0
    return meta.values


def train_meta_learner(oof_lin, oof_trans, y_train, meta_features_train):
    """
    Trains an XGBoost meta-learner on OOF predictions and meta-features.
    """
    print("Training XGBoost Meta-Learner...")

    # Concatenate inputs: [Linear Probs (3), Transformer Probs (3), Meta Features (2)]
    X_train = np.hstack([oof_lin, oof_trans, meta_features_train])

    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Train
    model = xgb.train(
        Config.XGB_PARAMS,
        dtrain,
        num_boost_round=Config.XGB_PARAMS["n_estimators"],
        verbose_eval=False,
    )

    return model


def predict_meta_learner(model, preds_lin, preds_trans, meta_features):
    """
    Generates predictions using the trained meta-learner.
    """
    X = np.hstack([preds_lin, preds_trans, meta_features])
    dtest = xgb.DMatrix(X)
    return model.predict(dtest)


import pandas as pd  # Ensure pandas is imported


def run_ensemble_stacking(
    train_oof_lin,
    val_pred_lin,
    test_pred_lin,
    train_oof_trans,
    val_pred_trans,
    test_pred_trans,
    y_train,
):
    """
    Orchestrates the Stacking Ensemble.
    """
    print("--- Starting Ensemble Pipeline (XGBoost Stacking) ---")

    # 1. Load Data for Meta-Features
    df_train = load_data("train")
    df_val = load_data("val")
    df_test = load_data("test")

    # 2. Extract Meta-Features
    meta_train = get_meta_features(df_train)
    meta_val = get_meta_features(df_val)
    meta_test = get_meta_features(df_test)

    # 3. Train Meta-Learner
    model = train_meta_learner(train_oof_lin, train_oof_trans, y_train, meta_train)

    # 4. Predict on Validation (for Final Metric)
    val_preds_final = predict_meta_learner(
        model, val_pred_lin, val_pred_trans, meta_val
    )

    # 5. Predict on Test (for Submission)
    test_preds_final = predict_meta_learner(
        model, test_pred_lin, test_pred_trans, meta_test
    )

    print("--- Ensemble Pipeline Complete ---")
    return val_preds_final, test_preds_final
