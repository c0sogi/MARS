import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.utils import set_seed


class LGBMRankerWrapper:
    def __init__(self, params=None):
        """
        Wrapper for LightGBM Ranker optimized for MAP@12.

        Args:
            params (dict, optional): Hyperparameters to override defaults.
        """
        set_seed(42)

        # Default parameters for ranking task
        self.params = {
            "objective": "lambdarank",
            "metric": "map",
            "eval_at": 12,
            "boosting_type": "gbdt",
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": 12,
        }

        if params:
            self.params.update(params)

        self.model = lgb.LGBMRanker(**self.params)
        self.is_fitted = False

    def fit(
        self, X_train, y_train, group_train, X_val=None, y_val=None, group_val=None
    ):
        """
        Trains the LightGBM Ranker model with early stopping.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.array): Training labels (relevance).
            group_train (np.array): Group sizes for training data (queries).
            X_val (pd.DataFrame, optional): Validation features.
            y_val (np.array, optional): Validation labels.
            group_val (np.array, optional): Group sizes for validation data.
        """
        eval_set = []
        eval_group = []

        if X_val is not None and y_val is not None and group_val is not None:
            eval_set = [(X_val, y_val)]
            eval_group = [group_val]

        # Configure callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ]

        self.model.fit(
            X_train,
            y_train,
            group=group_train,
            eval_set=eval_set,
            eval_group=eval_group,
            eval_at=[12],
            callbacks=callbacks,
        )
        self.is_fitted = True

    def predict(self, X):
        """
        Predicts relevance scores for the input features.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            np.array: Predicted scores.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(X)

    def get_feature_importance(self):
        """
        Returns a DataFrame of feature importances.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        return pd.DataFrame(
            {
                "feature": self.model.feature_name_,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    def save_model(self, path):
        """
        Saves the underlying booster to a text file.

        Args:
            path (str): File path to save the model.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.booster_.save_model(path)
        print(f"Model saved to {path}")


def generate_submission(test_df, scores, sample_submission_path, output_path):
    """
    Generates the submission CSV file by sorting predicted candidates and
    merging with the full customer list.

    Args:
        test_df (pd.DataFrame): DataFrame containing 'customer_id' and 'article_id' candidates.
        scores (np.array): Predicted scores corresponding to rows in test_df.
        sample_submission_path (str): Path to the sample submission file.
        output_path (str): Path where the final submission CSV will be saved.
    """
    print("Generating submission file...")

    # Create a working copy with scores
    df = test_df[["customer_id", "article_id"]].copy()
    df["score"] = scores

    # Sort candidates by customer and score (descending)
    df = df.sort_values(["customer_id", "score"], ascending=[True, False])

    # Select top 12 candidates per customer
    df = df.groupby("customer_id").head(12)

    # Format article_id to 10-digit string (e.g., 123 -> "0000000123")
    df["article_id"] = df["article_id"].astype(str).str.zfill(10)

    # Aggregate article_ids into a space-separated string
    preds = (
        df.groupby("customer_id")["article_id"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )
    preds.columns = ["customer_id", "prediction"]

    # Load sample submission to ensure all customers are included in the correct order
    sample_sub = pd.read_csv(sample_submission_path, usecols=["customer_id"])

    # Merge predictions onto the full customer list
    submission = sample_sub.merge(preds, on="customer_id", how="left")

    # Fill missing predictions with empty string (though candidates should cover all active users)
    submission["prediction"] = submission["prediction"].fillna("")

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
