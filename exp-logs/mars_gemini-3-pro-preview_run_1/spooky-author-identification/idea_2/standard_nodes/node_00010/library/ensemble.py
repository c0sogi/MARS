import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from library.utils import compute_log_loss, get_uncertainty_features
from library.config import Config


def prepare_stacking_features(probs_list, meta_features):
    """
    Combines base model probabilities, uncertainty features, and meta-features.
    Cite solution_lesson_node_00008 & 00009.
    """
    # 1. Base Probabilities
    base_probs = np.hstack(probs_list)

    # 2. Uncertainty Features
    uncertainty = get_uncertainty_features(probs_list)

    # 3. Combine with Meta Features
    return np.hstack([base_probs, uncertainty, meta_features])


def train_stacking_model(val_probs_list, val_meta, y_val):
    """
    Trains an XGBoost meta-learner using K-Fold CV on the validation set
    to generate unbiased OOF predictions for scoring, then retrains on full val.
    """
    print("Training Stacking Meta-Learner (XGBoost)...")

    X_full = prepare_stacking_features(val_probs_list, val_meta)
    y_full = y_val

    # K-Fold for OOF Scoring
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)
    oof_preds = np.zeros((len(y_full), 3))

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_full, y_full)):
        X_train_fold, X_valid_fold = X_full[train_idx], X_full[valid_idx]
        y_train_fold, y_valid_fold = y_full[train_idx], y_full[valid_idx]

        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)

        model_fold = xgb.train(
            Config.XGB_PARAMS,
            dtrain,
            num_boost_round=1000,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        oof_preds[valid_idx] = model_fold.predict(dvalid)

    oof_loss = compute_log_loss(y_full, oof_preds, labels=[0, 1, 2])
    print(f"Stacking OOF Log Loss: {oof_loss:.6f}")

    # Retrain on full validation set for final submission
    print("Retraining meta-learner on full validation set...")
    dtrain_full = xgb.DMatrix(X_full, label=y_full)
    final_model = xgb.train(
        Config.XGB_PARAMS,
        dtrain_full,
        num_boost_round=(
            int(model_fold.best_iteration * 1.2)
            if hasattr(model_fold, "best_iteration")
            else 200
        ),
        verbose_eval=False,
    )

    return final_model, oof_loss, oof_preds


def predict_stacking(model, test_probs_list, test_meta):
    """
    Generates predictions using the trained meta-learner.
    """
    X_test = prepare_stacking_features(test_probs_list, test_meta)
    dtest = xgb.DMatrix(X_test)
    return model.predict(dtest)
