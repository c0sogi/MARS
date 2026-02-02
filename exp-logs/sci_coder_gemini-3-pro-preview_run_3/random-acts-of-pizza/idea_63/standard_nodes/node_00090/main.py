import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

# Ensure the current directory is in the path for imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, print_header
from library.data_factory import DataFactory
from library.features import FeaturePipeline
from library.engine import HybridEnsembleEngine


def main():
    # 1. Setup
    set_seed(Config.SEED)
    VAL_THRESHOLD = 0.7222984867326668

    # 2. Data Loading
    print_header("Loading Data")
    # Load Union Data (Train + Val merged) for the Engine
    df_union = DataFactory.load_union_data(load_cached_data=True)
    # Load Test Data
    df_test = DataFactory.load_test_data()

    # Load original validation metadata to identify hold-out set for final metric
    val_meta_path = Config.VAL_PATH
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Validation metadata not found at {val_meta_path}")
    df_val_holdout = pd.read_parquet(val_meta_path)
    val_ids = set(df_val_holdout[Config.ID_COL].values)

    # 3. Feature Engineering
    print_header("Feature Engineering")
    pipeline = FeaturePipeline(df_union, df_test)

    # Generate all feature sets
    X_meta_train, X_meta_test = pipeline.get_augmented_metadata(load_cached_data=True)
    X_lex_train, X_lex_test = pipeline.get_granular_lexical(load_cached_data=True)
    X_beh_train, X_beh_test = pipeline.get_behavioral_sparse(load_cached_data=True)
    X_sem_train, X_sem_test = pipeline.get_semantic_dense(load_cached_data=True)

    # Organize into dictionaries for the Engine
    X_train_dict = {
        "metadata": X_meta_train,
        "lexical": X_lex_train,
        "behavioral": X_beh_train,
        "semantic": X_sem_train,
    }

    X_test_dict = {
        "metadata": X_meta_test,
        "lexical": X_lex_test,
        "behavioral": X_beh_test,
        "semantic": X_sem_test,
    }

    y_train = df_union[Config.TARGET_COL]
    test_ids = df_test[Config.ID_COL].values

    # 4. Model Training & Inference
    print_header("Running Hybrid Ensemble Engine")
    engine = HybridEnsembleEngine(
        X_train_dict=X_train_dict,
        y_train=y_train,
        X_test_dict=X_test_dict,
        test_ids=test_ids,
        output_dir=Config.WORKING_DIR,
    )

    # This runs CV, trains L1/L2 models, retrains stable models, and predicts test set
    engine.train_cv_and_predict()

    # 5. Validation Assessment
    print_header("Validation Assessment")

    # Load OOF predictions (Level 1 outputs)
    oof_path = os.path.join(Config.WORKING_DIR, "oof_predictions.csv")
    if not os.path.exists(oof_path):
        raise FileNotFoundError("OOF predictions not found.")

    df_oof = pd.read_csv(oof_path)

    # Align OOF with Union Data to get Request IDs
    # df_oof index corresponds to df_union index
    df_oof[Config.ID_COL] = df_union[Config.ID_COL].values
    df_oof[Config.TARGET_COL] = df_union[Config.TARGET_COL].values

    # Filter for the Hold-out Validation Set
    df_val_preds = df_oof[df_oof[Config.ID_COL].isin(val_ids)].copy()

    if len(df_val_preds) == 0:
        raise ValueError("No validation samples found in OOF predictions.")

    # Load trained Meta-Learner
    meta_learner_path = os.path.join(
        Config.WORKING_DIR, "models", "meta_learner.joblib"
    )
    meta_learner = joblib.load(meta_learner_path)

    # Prepare Level 1 features for Meta-Learner
    # The meta-learner expects columns matching the base models
    base_model_cols = [
        c
        for c in df_val_preds.columns
        if c not in [Config.ID_COL, Config.TARGET_COL, "Unnamed: 0"]
    ]
    X_val_meta = df_val_preds[base_model_cols]
    y_val_true = df_val_preds[Config.TARGET_COL]

    # Predict using Meta-Learner
    y_val_pred = meta_learner.predict_proba(X_val_meta)[:, 1]

    # Compute Metric
    final_metric = roc_auc_score(y_val_true, y_val_pred)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print_header("Failure Analysis")

    # Calculate Error
    errors = np.abs(y_val_true.values - y_val_pred)

    # Merge errors back to original validation metadata to access features
    df_analysis = df_val_holdout.copy()
    # Ensure alignment (df_val_preds might be shuffled relative to df_val_holdout, so merge on ID)
    error_df = pd.DataFrame(
        {Config.ID_COL: df_val_preds[Config.ID_COL], "error": errors}
    )
    df_analysis = df_analysis.merge(error_df, on=Config.ID_COL, how="inner")

    # Select numerical columns for correlation
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
    correlations = []

    for col in numeric_cols:
        if col == "error" or col == Config.TARGET_COL:
            continue
        # Handle NaNs just in case
        series = df_analysis[col].fillna(0)
        if series.std() == 0:
            continue

        corr = np.corrcoef(series, df_analysis["error"])[0, 1]
        if not np.isnan(corr):
            correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 7. Submission Management
    if final_metric <= VAL_THRESHOLD:
        print(
            f"\nMetric {final_metric} <= Threshold {VAL_THRESHOLD}. Removing submission file."
        )
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric {final_metric} > Threshold {VAL_THRESHOLD}. Submission retained."
        )


if __name__ == "__main__":
    main()
