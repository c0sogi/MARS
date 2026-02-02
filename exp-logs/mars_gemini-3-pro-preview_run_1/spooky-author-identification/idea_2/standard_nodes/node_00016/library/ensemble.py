import numpy as np
import xgboost as xgb
import pandas as pd
import numpy as np
from library.utils import compute_log_loss, generate_meta_features
from library.config import Config


def train_stacking_model(
    oof_linear,
    oof_transformer,
    y_train,
    val_linear,
    val_transformer,
    y_val,
    train_text,
    val_text,
):
    """
    Trains an XGBoost meta-learner using OOF predictions and meta-features.
    """
    print("Training XGBoost Meta-Learner...")

    # 1. Generate Meta Features
    print("Generating meta-features...")
    X_meta_train = generate_meta_features(train_text, [oof_linear, oof_transformer])
    X_meta_val = generate_meta_features(val_text, [val_linear, val_transformer])

    # 2. Prepare Input Data (Concatenate Probabilities + Meta Features)
    # Convert probs to DataFrames
    cols = ["lin_0", "lin_1", "lin_2", "trans_0", "trans_1", "trans_2"]

    X_train_probs = np.hstack([oof_linear, oof_transformer])
    X_val_probs = np.hstack([val_linear, val_transformer])

    X_train_df = pd.DataFrame(X_train_probs, columns=cols)
    X_val_df = pd.DataFrame(X_val_probs, columns=cols)

    # Concatenate
    X_train_full = pd.concat([X_train_df, X_meta_train], axis=1)
    X_val_full = pd.concat([X_val_df, X_meta_val], axis=1)

    # 3. Train XGBoost
    dtrain = xgb.DMatrix(X_train_full, label=y_train)
    dval = xgb.DMatrix(X_val_full, label=y_val)

    model = xgb.train(
        Config.XGB_PARAMS,
        dtrain,
        num_boost_round=Config.XGB_PARAMS["n_estimators"],
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=Config.XGB_PARAMS["early_stopping_rounds"],
        verbose_eval=50,
    )

    return model


def predict_stacking(model, test_linear, test_transformer, test_text):
    """
    Generates predictions using the trained meta-learner.
    """
    X_meta_test = generate_meta_features(test_text, [test_linear, test_transformer])

    cols = ["lin_0", "lin_1", "lin_2", "trans_0", "trans_1", "trans_2"]
    X_test_probs = np.hstack([test_linear, test_transformer])
    X_test_df = pd.DataFrame(X_test_probs, columns=cols)

    X_test_full = pd.concat([X_test_df, X_meta_test], axis=1)

    dtest = xgb.DMatrix(X_test_full)
    return model.predict(dtest)
