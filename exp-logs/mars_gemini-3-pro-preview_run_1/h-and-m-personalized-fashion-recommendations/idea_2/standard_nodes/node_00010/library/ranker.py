import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.utils import set_seed


class LGBMRankerWrapper:
    def __init__(self, params=None):
        """
        Heuristic Ranker replacing LightGBM.
        Cite solution_lesson_node_00009: Explicit heuristic cascades often outperform basic LTR models.
        """
        set_seed(42)
        self.is_fitted = False

    def fit(
        self, X_train, y_train, group_train, X_val=None, y_val=None, group_val=None
    ):
        """
        No-op for heuristic ranker.
        """
        print("Heuristic Ranker: No training required.")
        self.is_fitted = True

    def predict(self, X):
        """
        Predicts relevance scores using a strict heuristic cascade.
        Cite solution_lesson_node_00003: Prioritizes Repurchase > CF > Popularity.
        """
        scores = np.zeros(len(X))

        # Apply weights to enforce strict cascade
        # Repurchase scores are ~0.08 to 1.0. Multiplier 1e6 puts them in 80k - 1M range.
        if "repurchase_score" in X.columns:
            scores += X["repurchase_score"].values * 1_000_000

        # CF scores are dot products (usually < 100). Multiplier 1e3 puts them in < 100k range.
        if "cf_score" in X.columns:
            scores += X["cf_score"].values * 1_000

        # Pop scores are 0-1. No multiplier needed (or 1).
        if "pop_score" in X.columns:
            scores += X["pop_score"].values

        return scores

    def get_feature_importance(self):
        """
        Returns dummy importance.
        """
        return pd.DataFrame(
            {
                "feature": ["heuristic_cascade"],
                "importance": [1.0],
            }
        )

    def save_model(self, path):
        """
        No-op.
        """
        pass


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
