import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from library.config import SEED, SUBMISSION_DIR, WORK_DIR, META_MODEL_ALPHA
from library.utils import seed_everything
from library.data_factory import get_tabular_dataset, get_vision_dataset
from library.training_tabular import run_lgbm_cv
from library.training_vision import run_vision_cv
from library.training_meta import train_ridge_stack


def main():
    # 1. Setup
    seed_everything(SEED)

    # 2. Data Preparation
    # Trigger feature generation for all splits
    # We load them to ensure caches are built
    get_tabular_dataset("train", load_cached_data=True)
    get_tabular_dataset("val", load_cached_data=True)
    get_tabular_dataset("test", load_cached_data=True)

    get_vision_dataset("train", load_cached_data=True)
    get_vision_dataset("val", load_cached_data=True)
    get_vision_dataset("test", load_cached_data=True)

    # 3. Model Training
    # Run Tabular Branch (LightGBM)
    # Returns OOF and Test predictions
    oof_lgb, test_lgb = run_lgbm_cv(load_cached_data=True)

    # Run Vision Branch (EfficientNet)
    # Returns OOF and Test predictions
    oof_cnn, test_cnn = run_vision_cv(load_cached_data=True)

    # 4. Meta-Model Validation & Metric Calculation
    # Merge OOF DataFrames on segment_id and target
    # oof_lgb columns: segment_id, time_to_eruption, lgb_pred
    # oof_cnn columns: segment_id, time_to_eruption, cnn_pred
    train_stack = pd.merge(
        oof_lgb, oof_cnn[["segment_id", "cnn_pred"]], on="segment_id", how="inner"
    )

    X_train = train_stack[["lgb_pred", "cnn_pred"]].values
    y_train = train_stack["time_to_eruption"].values

    # Fit Ridge Regression (Meta-Learner)
    meta_model = Ridge(alpha=META_MODEL_ALPHA, random_state=SEED)
    meta_model.fit(X_train, y_train)

    # Predict on OOF to get final validation score
    oof_preds_meta = meta_model.predict(X_train)
    # Clip negative predictions
    oof_preds_meta = np.maximum(oof_preds_meta, 0)

    # Calculate MAE
    final_mae = mean_absolute_error(y_train, oof_preds_meta)
    print(f"Final Validation Metric: {final_mae}")

    # 5. Failure Analysis
    # Calculate absolute error
    train_stack["abs_error"] = np.abs(train_stack["time_to_eruption"] - oof_preds_meta)

    # Load original tabular features for correlation analysis
    # We need to combine train and val splits to match the OOF set
    df_train_feat = get_tabular_dataset("train", load_cached_data=True)
    df_val_feat = get_tabular_dataset("val", load_cached_data=True)
    df_features = pd.concat([df_train_feat, df_val_feat], axis=0).reset_index(drop=True)

    # Merge features with errors
    df_analysis = pd.merge(
        df_features,
        train_stack[["segment_id", "abs_error"]],
        on="segment_id",
        how="inner",
    )

    # Calculate correlations between features and absolute error
    # Select only numeric columns
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns
    # Drop target and error itself from correlation
    cols_to_exclude = ["time_to_eruption", "abs_error", "segment_id"]
    cols_to_corr = [c for c in numeric_cols if c not in cols_to_exclude]

    if cols_to_corr:
        correlations = (
            df_analysis[cols_to_corr]
            .corrwith(df_analysis["abs_error"])
            .sort_values(ascending=False)
        )
        print("\nFailure Analysis - Top 5 Features Correlated with Error:")
        print(correlations.head(5))

    # 6. Submission Generation
    # Threshold check
    THRESHOLD = 2250276.65

    if final_mae < THRESHOLD:
        train_ridge_stack(oof_lgb, test_lgb, oof_cnn, test_cnn)
    else:
        print(
            f"\nValidation Metric ({final_mae}) is not lower than threshold ({THRESHOLD}). Submission skipped."
        )
        # Ensure no submission file exists
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        if os.path.exists(submission_path):
            os.remove(submission_path)


if __name__ == "__main__":
    main()
