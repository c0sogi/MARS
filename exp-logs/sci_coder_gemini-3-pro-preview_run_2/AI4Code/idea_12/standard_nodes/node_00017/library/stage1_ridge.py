import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from library.config import Config


class RidgeStacker:
    """
    Implements the Stage 1 Ridge Regression model for the stacking pipeline.
    This model serves as a high-bias, low-variance baseline ("Signpost" model)
    mapping TF-IDF vectors directly to normalized ranks.
    """

    def __init__(self):
        """
        Initialize the RidgeStacker with configuration parameters.
        """
        self.model_path = Config.CACHE_STAGE1_RIDGE
        self.model = None
        self.alpha = Config.RIDGE_ALPHA
        self.solver = Config.RIDGE_SOLVER
        self.random_state = Config.SEED

    def train_and_predict_oof(self, df: pd.DataFrame, text_pipeline) -> pd.DataFrame:
        """
        Trains Ridge Regression using GroupKFold cross-validation to generate
        unbiased Out-Of-Fold (OOF) predictions for the training set.

        Args:
            df (pd.DataFrame): DataFrame containing training data.
                               Must contain 'cell_type', 'source', 'pct_rank', 'ancestor_id'.
            text_pipeline: Fitted TextPipeline instance for vectorization.

        Returns:
            pd.DataFrame: DataFrame containing ['cell_id', 'ridge_pred'] for markdown cells.
        """
        # Filter for markdown cells only
        md_mask = df["cell_type"] == "markdown"
        df_md = df[md_mask].copy()

        if len(df_md) == 0:
            return pd.DataFrame(columns=["cell_id", "ridge_pred"])

        print(f"Stage 1: Generating OOF predictions for {len(df_md)} markdown cells...")

        # Transform text using the provided pipeline
        # TextPipeline.transform returns (X_tfidf, X_svd). We only need X_tfidf for Ridge.
        X_tfidf, _ = text_pipeline.transform(df_md["source"].tolist())
        y = df_md["pct_rank"].values
        groups = df_md["ancestor_id"].values
        cell_ids = df_md["cell_id"].values

        # Initialize arrays
        oof_preds = np.zeros(len(df_md))

        # 5-Fold Group CV
        # We use GroupKFold to ensure notebooks with the same ancestor are in the same fold
        gkf = GroupKFold(n_splits=5)

        fold = 1
        for train_idx, val_idx in gkf.split(X_tfidf, y, groups):
            X_train, y_train = X_tfidf[train_idx], y[train_idx]
            X_val, y_val = X_tfidf[val_idx], y[val_idx]

            # Initialize and train model
            model = Ridge(
                alpha=self.alpha, solver=self.solver, random_state=self.random_state
            )
            model.fit(X_train, y_train)

            # Predict
            val_preds = model.predict(X_val)
            oof_preds[val_idx] = val_preds

            # Calculate and print metric
            mae = np.mean(np.abs(y_val - val_preds))
            print(f"Fold {fold} MAE: {mae}")
            fold += 1

        total_mae = np.mean(np.abs(y - oof_preds))
        print(f"Stage 1 OOF Mean MAE: {total_mae}")

        return pd.DataFrame({"cell_id": cell_ids, "ridge_pred": oof_preds})

    def fit(self, df: pd.DataFrame, text_pipeline):
        """
        Trains the Ridge model on the full provided dataset and saves it to disk.

        Args:
            df (pd.DataFrame): DataFrame containing training data.
            text_pipeline: Fitted TextPipeline instance.
        """
        md_mask = df["cell_type"] == "markdown"
        df_md = df[md_mask]

        if len(df_md) == 0:
            print("Warning: No markdown cells found to fit Stage 1 model.")
            return

        print(f"Stage 1: Fitting final Ridge model on {len(df_md)} markdown cells...")

        X_tfidf, _ = text_pipeline.transform(df_md["source"].tolist())
        y = df_md["pct_rank"].values

        self.model = Ridge(
            alpha=self.alpha, solver=self.solver, random_state=self.random_state
        )
        self.model.fit(X_tfidf, y)

        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Stage 1 Ridge model saved to {self.model_path}")

    def predict(self, df: pd.DataFrame, text_pipeline) -> pd.DataFrame:
        """
        Generates predictions using the trained Ridge model.
        Loads the model from disk if not already loaded.

        Args:
            df (pd.DataFrame): DataFrame containing data to predict.
            text_pipeline: Fitted TextPipeline instance.

        Returns:
            pd.DataFrame: DataFrame containing ['cell_id', 'ridge_pred'] for markdown cells.
        """
        # Load model if needed
        if self.model is None:
            if os.path.exists(self.model_path):
                print(f"Loading Stage 1 Ridge model from {self.model_path}")
                self.model = joblib.load(self.model_path)
            else:
                raise FileNotFoundError(
                    f"Stage 1 model not found at {self.model_path}. Call fit() first."
                )

        md_mask = df["cell_type"] == "markdown"
        df_md = df[md_mask]

        if len(df_md) == 0:
            return pd.DataFrame(columns=["cell_id", "ridge_pred"])

        X_tfidf, _ = text_pipeline.transform(df_md["source"].tolist())
        preds = self.model.predict(X_tfidf)

        return pd.DataFrame({"cell_id": df_md["cell_id"].values, "ridge_pred": preds})
