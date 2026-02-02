import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from library.config import Config
from library.utils import seed_everything, mae_score


class RidgeStacker:
    """
    Implements the Meta-Learner using Ridge Regression.
    Learns to blend predictions from the Tabular and Vision branches.
    """

    def __init__(self, alpha=Config.RIDGE_ALPHA, random_state=Config.SEED):
        self.alpha = alpha
        self.random_state = random_state
        self.model = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.feature_cols = ["pred_tabular", "pred_vision"]

    def fit(self, X, y):
        """
        Trains the Ridge regressor.

        Args:
            X (pd.DataFrame): Feature matrix with columns ['pred_tabular', 'pred_vision'].
            y (pd.Series or np.array): Target values.
        """
        self.model.fit(X[self.feature_cols], y)

    def predict(self, X):
        """
        Generates predictions using the trained model.

        Args:
            X (pd.DataFrame): Feature matrix with columns ['pred_tabular', 'pred_vision'].

        Returns:
            np.array: Predicted values.
        """
        preds = self.model.predict(X[self.feature_cols])
        # Clip negative predictions to 0 as time cannot be negative
        return np.maximum(preds, 0)


def run_stacking(tabular_oof, tabular_test, vision_oof, vision_test, ground_truth_df):
    """
    Orchestrates the stacking pipeline:
    1. Aligns OOF predictions from both branches with Ground Truth.
    2. Trains the Ridge Meta-Learner.
    3. Aligns Test predictions.
    4. Generates final submission.

    Args:
        tabular_oof (pd.DataFrame): OOF preds from Branch A. Cols: ['segment_id', 'time_to_eruption']
        tabular_test (pd.DataFrame): Test preds from Branch A. Cols: ['segment_id', 'time_to_eruption']
        vision_oof (pd.DataFrame): OOF preds from Branch B. Cols: ['segment_id', 'time_to_eruption_pred']
        vision_test (pd.DataFrame): Test preds from Branch B. Cols: ['segment_id', 'time_to_eruption']
        ground_truth_df (pd.DataFrame): Metadata with actual targets. Cols: ['segment_id', 'time_to_eruption']

    Returns:
        pd.DataFrame: Final submission DataFrame.
    """
    seed_everything(Config.SEED)

    print("--- Starting Meta-Learner Stacking ---")

    # ---------------------------------------------------------
    # 1. Prepare Training Data (OOF)
    # ---------------------------------------------------------
    # Rename columns to avoid collision and ensure clarity
    # Tabular OOF usually comes with 'time_to_eruption' as the prediction column name (mimicking submission format)
    tab_oof_clean = tabular_oof.rename(columns={"time_to_eruption": "pred_tabular"})[
        ["segment_id", "pred_tabular"]
    ]

    # Vision OOF usually comes with 'time_to_eruption_pred'
    # Check column existence to be safe, as naming might vary slightly in implementation
    vis_col = (
        "time_to_eruption_pred"
        if "time_to_eruption_pred" in vision_oof.columns
        else "time_to_eruption"
    )
    vis_oof_clean = vision_oof.rename(columns={vis_col: "pred_vision"})[
        ["segment_id", "pred_vision"]
    ]

    # Merge predictions
    train_stack = pd.merge(tab_oof_clean, vis_oof_clean, on="segment_id", how="inner")

    # Merge with Ground Truth
    # ground_truth_df contains the actual 'time_to_eruption'
    train_stack = pd.merge(
        train_stack,
        ground_truth_df[["segment_id", "time_to_eruption"]],
        on="segment_id",
        how="inner",
    )

    X_train = train_stack[["pred_tabular", "pred_vision"]]
    y_train = train_stack["time_to_eruption"]

    print(f"Training Meta-Learner on {len(train_stack)} aligned OOF samples...")

    # ---------------------------------------------------------
    # 2. Train Meta-Learner
    # ---------------------------------------------------------
    stacker = RidgeStacker()
    stacker.fit(X_train, y_train)

    # Evaluate on OOF
    oof_preds = stacker.predict(X_train)
    final_mae = mae_score(y_train, oof_preds)
    print(f"Final Ensemble OOF MAE: {final_mae}")

    # Print learned coefficients to see contribution of each branch
    coefs = stacker.model.coef_
    intercept = stacker.model.intercept_
    print(
        f"Meta-Learner Weights -> Tabular: {coefs[0]:.4f}, Vision: {coefs[1]:.4f}, Intercept: {intercept:.4f}"
    )

    # ---------------------------------------------------------
    # 3. Prepare Test Data & Inference
    # ---------------------------------------------------------
    # Rename columns
    tab_test_clean = tabular_test.rename(columns={"time_to_eruption": "pred_tabular"})[
        ["segment_id", "pred_tabular"]
    ]

    vis_col_test = "time_to_eruption"  # Test output usually follows submission format
    vis_test_clean = vision_test.rename(columns={vis_col_test: "pred_vision"})[
        ["segment_id", "pred_vision"]
    ]

    # Merge
    test_stack = pd.merge(tab_test_clean, vis_test_clean, on="segment_id", how="inner")

    print(f"Generating predictions for {len(test_stack)} test samples...")

    final_test_preds = stacker.predict(test_stack)

    # ---------------------------------------------------------
    # 4. Create Submission File
    # ---------------------------------------------------------
    submission_df = pd.DataFrame(
        {"segment_id": test_stack["segment_id"], "time_to_eruption": final_test_preds}
    )

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    save_path = Config.SUBMISSION_PATH
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

    return submission_df
