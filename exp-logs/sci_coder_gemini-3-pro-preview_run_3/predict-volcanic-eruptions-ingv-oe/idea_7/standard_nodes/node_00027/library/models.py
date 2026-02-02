import lightgbm as lgb
import xgboost as xgb


def get_lightgbm_regressor(
    random_state=42, n_estimators=10000, learning_rate=0.01, n_jobs=12
):
    """
    Returns a LightGBM Regressor configured for Mean Absolute Error minimization.

    Key Hyperparameters:
    - objective: 'regression_l1' (MAE)
    - n_estimators: High count (default 10000) for extended training with early stopping.
    - learning_rate: Low (0.01) for robust convergence.
    - Regularization: Uses subsampling and column sampling to prevent overfitting on the wide feature set.
    """
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=63,  # Allows for capturing complex non-linearities
        max_depth=-1,  # Depth controlled by num_leaves
        min_child_samples=20,
        subsample=0.8,  # Bagging fraction
        subsample_freq=1,  # Perform bagging every k iterations
        colsample_bytree=0.8,  # Feature fraction per tree
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1,  # L2 regularization
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1,  # Suppress warnings and info
        importance_type="gain",
    )
    return model


def get_xgboost_regressor(
    random_state=42, n_estimators=10000, learning_rate=0.01, n_jobs=12, device="cpu"
):
    """
    Returns an XGBoost Regressor configured for Mean Absolute Error minimization.

    Key Hyperparameters:
    - objective: 'reg:absoluteerror' (MAE)
    - tree_method: 'hist' for efficient training on large tabular datasets.
    - Regularization: Includes Gamma, Alpha, and Lambda to control model complexity.
    """
    model = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=8,  # Moderate depth to capture interactions
        min_child_weight=1,
        gamma=0.1,  # Minimum loss reduction required for partition
        subsample=0.8,  # Row subsampling
        colsample_bytree=0.8,  # Column subsampling
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1,  # L2 regularization
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,  # Silent mode
        tree_method="hist",  # Histogram-based algorithm (fast)
        device=device,  # Support for 'cuda' if GPU is available
    )
    return model
