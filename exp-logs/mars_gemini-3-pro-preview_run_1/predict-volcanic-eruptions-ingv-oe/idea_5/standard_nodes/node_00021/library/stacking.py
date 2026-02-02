import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from library.config import Config
from library.utils import seed_everything


def train_meta_learner(df_oof_tab, df_test_tab, df_oof_vis, df_test_vis):
    """
    Trains a Ridge Regression meta-learner on the OOF predictions from the
    Tabular (Branch A) and Vision (Branch B) models, and generates the final
    submission file.

    Args:
        df_oof_tab (pd.DataFrame): OOF predictions from Tabular model.
                                   Cols: [segment_id, pred_time_to_eruption, true_time_to_eruption]
        df_test_tab (pd.DataFrame): Test predictions from Tabular model.
                                    Cols: [segment_id, time_to_eruption]
        df_oof_vis (pd.DataFrame): OOF predictions from Vision model.
                                   Cols: [segment_id, pred_time_to_eruption, true_time_to_eruption]
        df_test_vis (pd.DataFrame): Test predictions from Vision model.
                                    Cols: [segment_id, time_to_eruption]

    Returns:
        None
    """
    seed_everything(Config.SEED)

    print("Initializing Meta-Learner (Stacking)...")

    # ---------------------------------------------------------
    # 1. Prepare OOF Data for Meta-Training
    # ---------------------------------------------------------
    # Rename columns to avoid collision and ensure clarity
    oof_tab = df_oof_tab.rename(columns={"pred_time_to_eruption": "pred_tab"})
    oof_vis = df_oof_vis.rename(columns={"pred_time_to_eruption": "pred_vis"})

    # Merge on segment_id
    # We assume true_time_to_eruption is identical in both, but we keep one
    df_meta_train = pd.merge(
        oof_tab[["segment_id", "pred_tab", "true_time_to_eruption"]],
        oof_vis[["segment_id", "pred_vis"]],
        on="segment_id",
        how="inner",
    )

    # Define Features and Target
    X_meta = df_meta_train[["pred_tab", "pred_vis"]]
    y_meta = df_meta_train["true_time_to_eruption"]

    print(f"Meta-Learner Training Data Shape: {X_meta.shape}")

    # ---------------------------------------------------------
    # 2. Train Ridge Regressor
    # ---------------------------------------------------------
    meta_model = Ridge(alpha=Config.META_RIDGE_ALPHA, random_state=Config.SEED)
    meta_model.fit(X_meta, y_meta)

    # ---------------------------------------------------------
    # 3. Evaluate Meta-Learner
    # ---------------------------------------------------------
    # Predict on the training set (OOF) to see how well the blending works
    meta_oof_preds = meta_model.predict(X_meta)

    # Ensure non-negative predictions (time cannot be negative)
    meta_oof_preds = np.maximum(meta_oof_preds, 0)

    score = mean_absolute_error(y_meta, meta_oof_preds)

    print("\n--- Meta-Learner Evaluation ---")
    print(
        f"Coefficients: Tabular={meta_model.coef_[0]:.4f}, Vision={meta_model.coef_[1]:.4f}"
    )
    print(f"Intercept: {meta_model.intercept_:.4f}")
    print(f"Stacked OOF MAE: {score}")

    # ---------------------------------------------------------
    # 4. Generate Test Predictions
    # ---------------------------------------------------------
    # Rename columns
    test_tab = df_test_tab.rename(columns={"time_to_eruption": "pred_tab"})
    test_vis = df_test_vis.rename(columns={"time_to_eruption": "pred_vis"})

    # Merge
    df_meta_test = pd.merge(
        test_tab[["segment_id", "pred_tab"]],
        test_vis[["segment_id", "pred_vis"]],
        on="segment_id",
        how="inner",
    )

    X_test_meta = df_meta_test[["pred_tab", "pred_vis"]]

    # Predict
    final_test_preds = meta_model.predict(X_test_meta)

    # Clip negative values
    final_test_preds = np.maximum(final_test_preds, 0)

    # ---------------------------------------------------------
    # 5. Create Submission File
    # ---------------------------------------------------------
    submission = pd.DataFrame(
        {"segment_id": df_meta_test["segment_id"], "time_to_eruption": final_test_preds}
    )

    # Ensure segment_id is integer
    submission["segment_id"] = submission["segment_id"].astype(int)

    print(f"\nSaving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
