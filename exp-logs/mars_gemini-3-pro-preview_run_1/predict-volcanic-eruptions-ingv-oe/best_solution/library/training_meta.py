import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from library.config import (
    WORK_DIR,
    SUBMISSION_DIR,
    META_MODEL_ALPHA,
)
from library.utils import calc_mae


def train_ridge_stack(
    oof_tabular: pd.DataFrame = None,
    test_tabular: pd.DataFrame = None,
    oof_vision: pd.DataFrame = None,
    test_vision: pd.DataFrame = None,
):
    """
    Trains a Ridge Regression Meta-Learner to stack predictions from the Tabular and Vision branches.
    Generates the final submission file.

    Args:
        oof_tabular (pd.DataFrame, optional): OOF predictions from LightGBM.
        test_tabular (pd.DataFrame, optional): Test predictions from LightGBM.
        oof_vision (pd.DataFrame, optional): OOF predictions from CNN.
        test_vision (pd.DataFrame, optional): Test predictions from CNN.

    If arguments are None, attempts to load them from the working directory.
    """
    print("Initializing Meta-Learner (Ridge Regression Stacking)...")

    # ---------------------------------------------------------
    # 1. Load Data (if not provided)
    # ---------------------------------------------------------
    if oof_tabular is None:
        path = os.path.join(WORK_DIR, "lgbm_oof.csv")
        print(f"Loading Tabular OOF from {path}")
        oof_tabular = pd.read_csv(path)

    if test_tabular is None:
        path = os.path.join(WORK_DIR, "lgbm_test.csv")
        print(f"Loading Tabular Test from {path}")
        test_tabular = pd.read_csv(path)

    if oof_vision is None:
        path = os.path.join(WORK_DIR, "cnn_oof.csv")
        print(f"Loading Vision OOF from {path}")
        oof_vision = pd.read_csv(path)

    if test_vision is None:
        path = os.path.join(WORK_DIR, "cnn_test.csv")
        print(f"Loading Vision Test from {path}")
        test_vision = pd.read_csv(path)

    # ---------------------------------------------------------
    # 2. Merge Predictions
    # ---------------------------------------------------------
    # Merge OOF dataframes on segment_id and target
    # Ensure we align rows correctly
    train_stack = pd.merge(
        oof_tabular,
        oof_vision,
        on=["segment_id", "time_to_eruption"],
        how="inner",
        suffixes=("_lgb", "_cnn"),
    )

    # Handle column naming if suffixes didn't apply (e.g. if names were already distinct)
    # Expected cols: 'lgb_pred', 'cnn_pred'
    # If they were named 'lgb_pred' and 'cnn_pred' in input, merge keeps them.
    # If they were generic 'pred', suffixes apply.
    # Based on previous modules: Tabular saves 'lgb_pred', Vision saves 'cnn_pred'.

    # Verify columns exist
    if "lgb_pred" not in train_stack.columns or "cnn_pred" not in train_stack.columns:
        raise KeyError(
            "Expected columns 'lgb_pred' and 'cnn_pred' not found in merged OOF data."
        )

    # Merge Test dataframes
    test_stack = pd.merge(test_tabular, test_vision, on="segment_id", how="inner")

    print(f"Stacked Train Shape: {train_stack.shape}")
    print(f"Stacked Test Shape: {test_stack.shape}")

    # ---------------------------------------------------------
    # 3. Prepare Features and Targets
    # ---------------------------------------------------------
    X_train = train_stack[["lgb_pred", "cnn_pred"]].values
    y_train = train_stack["time_to_eruption"].values

    X_test = test_stack[["lgb_pred", "cnn_pred"]].values
    test_ids = test_stack["segment_id"].values

    # ---------------------------------------------------------
    # 4. Train Ridge Regression
    # ---------------------------------------------------------
    # We use a linear combination of the two models.
    # Fit intercept allows for bias correction.
    meta_model = Ridge(alpha=META_MODEL_ALPHA, random_state=42)
    meta_model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 5. Evaluation
    # ---------------------------------------------------------
    oof_preds_meta = meta_model.predict(X_train)

    # Clip negative predictions (physical impossibility)
    oof_preds_meta = np.maximum(oof_preds_meta, 0)

    mae = calc_mae(y_train, oof_preds_meta)

    print("\n--- Meta-Learner Results ---")
    print(
        f"Ridge Coefficients: LGBM={meta_model.coef_[0]:.4f}, CNN={meta_model.coef_[1]:.4f}"
    )
    print(f"Ridge Intercept: {meta_model.intercept_:.4f}")
    print(f"Stacked OOF MAE: {mae}")

    # ---------------------------------------------------------
    # 6. Generate Submission
    # ---------------------------------------------------------
    test_preds_meta = meta_model.predict(X_test)

    # Clip negative predictions
    test_preds_meta = np.maximum(test_preds_meta, 0)

    submission_df = pd.DataFrame(
        {"segment_id": test_ids, "time_to_eruption": test_preds_meta}
    )

    # Ensure output directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Submission Head:\n{submission_df.head()}")

    return submission_df
