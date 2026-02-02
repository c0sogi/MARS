import numpy as np
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from library.config import Config
from library.utils import setup_logger


class Level1Predictors:
    """
    Level 1 Base Learners for the Stacking Ensemble.
    Includes SVR (with PCA), LightGBM, and MLP (replacing Ridge).
    """

    def __init__(self):
        self.logger = setup_logger(name="level1_predictors")

        # 1. SVR Pipeline
        # PCA is used to reduce dimensionality before SVR to improve training speed and performance
        # StandardScaler is essential for SVR convergence
        self.svr_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=Config.SVR_PCA_COMPONENTS, random_state=Config.SEED
                    ),
                ),
                ("svr", SVR(**Config.SVR_PARAMS)),
            ]
        )

        # 2. LightGBM Regressor
        # We extract early_stopping_rounds to use in the fit callback
        self.lgbm_params = Config.LGBM_PARAMS.copy()
        self.lgbm_es_rounds = self.lgbm_params.pop("early_stopping_rounds", None)
        self.lgbm = lgb.LGBMRegressor(**self.lgbm_params)

        # 3. MLP Regressor (Replaces Ridge)
        # Using MLP to provide non-linear diversity in the ensemble (Cite Lesson 00011)
        self.mlp = MLPRegressor(**Config.MLP_PARAMS)

    def fit(self, X, y):
        """
        Fits all Level 1 models.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        # Fit SVR
        self.logger.info("Training Level 1 Model: SVR (with PCA)...")
        self.svr_pipeline.fit(X, y)

        # Fit MLP
        self.logger.info("Training Level 1 Model: MLP...")
        # Scale inputs for MLP (it is sensitive to scale)
        # Note: MLPRegressor has internal scaling if solver='lbfgs', but for 'adam' it expects scaled data.
        # However, we are passing raw features which might be large.
        # Ideally we should pipeline it, but for simplicity we rely on the fact that
        # our embeddings are somewhat normalized and metadata is binary.
        # To be safe, we wrap it in a pipeline locally or just fit.
        # Given the previous code didn't scale for Ridge, we'll rely on MLP's robustness or add scaling.
        # Let's add scaling to be safe as MLP requires it.
        self.mlp_pipeline = Pipeline([("scaler", StandardScaler()), ("mlp", self.mlp)])
        self.mlp_pipeline.fit(X, y)

        # Fit LightGBM
        self.logger.info("Training Level 1 Model: LightGBM...")

        # Create an internal validation split for Early Stopping
        # This allows us to use early stopping even if the external caller provides only 'train' data
        # We use a 90/10 split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=Config.SEED
        )

        callbacks = []
        if self.lgbm_es_rounds:
            # verbose=False to keep logs clean; we log the result manually
            callbacks.append(
                lgb.early_stopping(stopping_rounds=self.lgbm_es_rounds, verbose=False)
            )
            callbacks.append(lgb.log_evaluation(period=0))  # Disable verbose logging

        self.lgbm.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=callbacks,
        )

        # Log LightGBM performance
        if self.lgbm.best_score_:
            # Retrieve RMSE from the validation set
            valid_rmse = self.lgbm.best_score_.get("valid_0", {}).get("rmse", "N/A")
            self.logger.info(f"LightGBM Best Validation RMSE: {valid_rmse}")

    def predict(self, X):
        """
        Generates predictions from all Level 1 models.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Matrix of shape (N, 3) containing predictions from SVR, LGBM, MLP.
        """
        pred_svr = self.svr_pipeline.predict(X)
        pred_lgbm = self.lgbm.predict(X)
        pred_mlp = self.mlp_pipeline.predict(X)

        # Stack predictions column-wise: [SVR, LGBM, MLP]
        return np.column_stack((pred_svr, pred_lgbm, pred_mlp))


class MetaLearner:
    """
    Level 2 Meta-Learner.
    Uses Linear Regression to combine Level 1 predictions.
    """

    def __init__(self):
        self.logger = setup_logger(name="meta_learner")
        self.model = LinearRegression()

    def fit(self, X_level1, y):
        """
        Fits the meta-learner.

        Args:
            X_level1 (np.ndarray): Predictions from Level 1 models (N, 3).
            y (np.ndarray): Target vector.
        """
        self.logger.info("Training Level 2 Meta-Learner (Linear Regression)...")
        self.model.fit(X_level1, y)

        # Log coefficients to understand the contribution of each base model
        # Order corresponds to predict stack: SVR, LGBM, MLP
        self.logger.info(
            f"Meta-Learner Coefficients (SVR, LGBM, MLP): {self.model.coef_}"
        )
        self.logger.info(f"Meta-Learner Intercept: {self.model.intercept_}")

    def predict(self, X_level1):
        """
        Generates final predictions.

        Args:
            X_level1 (np.ndarray): Predictions from Level 1 models (N, 3).

        Returns:
            np.ndarray: Final predictions.
        """
        return self.model.predict(X_level1)
