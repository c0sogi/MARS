import os
import sys
import numpy as np
import pandas as pd
import random
import warnings
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.data_manager import DataManager
from library.feature_engine import FeatureEngine
from library.stage1_model import RidgeStacker
from library.stage2_model import LGBMRanker
from library.inference import SubmissionGenerator
from library.utils import kendall_tau


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(Config.RANDOM_STATE)
    Config.setup()
    warnings.filterwarnings("ignore")

    # 2. Data Loading
    dm = DataManager()

    # Load metadata to get IDs for subsampling
    df_train_meta, df_val_meta, _ = dm.load_metadata()

    # Subsample Training Data for Fast Baseline (10,000 notebooks)
    unique_train_ids = df_train_meta["id"].unique()
    TARGET_TRAIN_SIZE = 10000

    if len(unique_train_ids) > TARGET_TRAIN_SIZE:
        sampled_ids = np.random.choice(
            unique_train_ids, size=TARGET_TRAIN_SIZE, replace=False
        )
        df_train_meta_sampled = df_train_meta[
            df_train_meta["id"].isin(sampled_ids)
        ].copy()
    else:
        df_train_meta_sampled = df_train_meta.copy()

    # Load actual cell data
    # We load the full cached dataframe (if available) and then filter
    df_train_full = dm.get_train_data(load_cached_data=True)
    df_train = df_train_full[
        df_train_full["id"].isin(df_train_meta_sampled["id"])
    ].reset_index(drop=True)

    df_val = dm.get_val_data(load_cached_data=True)

    # 3. Feature Engineering
    fe = FeatureEngine()

    # Fit on the sampled training data
    fe.fit(df_train)

    # Transform
    # Using specific names to manage caching for this run
    X_train_sparse, df_train_feats = fe.transform(
        df_train, name="train_baseline", load_cached_data=True
    )
    X_val_sparse, df_val_feats = fe.transform(
        df_val, name="val_baseline", load_cached_data=True
    )

    # 4. Stage 1: Ridge Regression
    s1 = RidgeStacker()

    # Prepare targets and groups
    y_train = df_train_feats["rank"].values
    groups = df_train_feats["ancestor_id"].values

    # Train OOF
    oof_preds = s1.train_oof(
        X_train_sparse, y_train, groups, n_splits=5, load_cached_data=False
    )
    df_train_feats["pred_ridge"] = oof_preds

    # Fit Final Model
    s1.fit_final(X_train_sparse, y_train)

    # Predict on Validation
    val_ridge_preds = s1.predict(X_val_sparse)
    df_val_feats["pred_ridge"] = val_ridge_preds

    # 5. Stage 2: LightGBM
    s2 = LGBMRanker()

    # Define features
    feature_cols = [
        "lex_mean",
        "lex_std",
        "lat_mean",
        "lat_std",
        "sym_mean",
        "sym_std",
        "md_ratio",
        "total_code",
        "pred_ridge",
    ]
    # Add SVD columns
    for i in range(Config.SVD_COMPONENTS):
        feature_cols.append(f"svd_{i}")

    s2.train(df_train_feats, df_val_feats, feature_cols)

    # Predict on Validation
    val_lgbm_preds = s2.predict(df_val_feats, feature_cols)
    df_val_feats["pred_rank"] = val_lgbm_preds

    # 6. Validation & Metric Calculation

    # Reconstruct the order for each notebook in validation set
    # 1. Get Code Cells with their implicit ranks (0.0 to 1.0 based on position)
    df_val_code = df_val[df_val["cell_type"] == "code"].copy()

    code_counts = df_val_code.groupby("id")["cell_id"].transform("count")
    code_cumcounts = df_val_code.groupby("id").cumcount()
    df_val_code["pred_rank"] = np.where(
        code_counts > 1, code_cumcounts / (code_counts - 1), 0.0
    )

    # 2. Get Markdown Cells with predicted ranks
    df_val_md = df_val_feats[["id", "cell_id", "pred_rank"]].copy()

    # 3. Combine
    df_val_pred_all = pd.concat(
        [df_val_md, df_val_code[["id", "cell_id", "pred_rank"]]], ignore_index=True
    )

    # 4. Sort
    df_val_pred_all.sort_values(by=["id", "pred_rank"], inplace=True)

    # 5. Aggregate to list
    val_preds_map = df_val_pred_all.groupby("id")["cell_id"].apply(list).to_dict()

    # 6. Get Ground Truth
    val_gt_map = dict(zip(df_val_meta["id"], df_val_meta["cell_order"].str.split()))

    # 7. Compute Kendall Tau
    gt_list = []
    pred_list = []

    common_ids = set(val_gt_map.keys()) & set(val_preds_map.keys())
    for nid in common_ids:
        gt_list.append(val_gt_map[nid])
        pred_list.append(val_preds_map[nid])

    score = kendall_tau(gt_list, pred_list)

    print(f"Final Validation Metric: {score}")

    # 7. Failure Analysis
    print("Running Failure Analysis...")
    # Merge GT rank back to preds to calc error
    df_analysis = df_val_feats.copy()
    df_analysis["error"] = (df_analysis["pred_rank"] - df_analysis["rank"]).abs()

    analysis_cols = ["lex_std", "lat_std", "sym_std", "md_ratio", "total_code"]
    print("Correlation between Error and Features:")
    for col in analysis_cols:
        if col in df_analysis.columns:
            corr, _ = pearsonr(df_analysis["error"], df_analysis[col])
            print(f"  Error vs {col}: {corr:.4f}")

    # 8. Submission
    THRESHOLD = 0.7959051868218839
    if score > THRESHOLD:
        print("Metric above threshold. Generating Submission...")
        sub_gen = SubmissionGenerator()
        sub_gen.generate_submission(load_cached_data=True)
    else:
        print(f"Metric {score} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
