import os
import sys
import numpy as np
import pandas as pd
import random
import joblib
import lightgbm as lgb
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(".")

# Import from provided library files
from library.config import Config
from library.data_manager import DataManager
from library.feature_engine import FeatureEngineer
from library.modeling import StackedRanker
from library.utils import kendall_tau_metric, convert_ranks_to_order


def set_seeds(seed=42):
    """Sets fixed seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seeds(Config.RANDOM_STATE)

    # Ensure output directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("Initializing Pipeline...")
    data_manager = DataManager()
    feature_engineer = FeatureEngineer()
    ranker = StackedRanker()

    # 2. Data Loading & Feature Engineering (Train)
    print("\n=== Processing Training Data ===")
    # Load raw training data
    df_train_raw = data_manager.load_data(split="train", load_cached_data=True)

    # Generate training features (this fits the vectorizers)
    df_train_features = feature_engineer.generate_features(
        df_train_raw, split="train", load_cached_data=True
    )

    # 3. Data Loading & Feature Engineering (Validation)
    print("\n=== Processing Validation Data ===")
    df_val_raw = data_manager.load_data(split="val", load_cached_data=True)
    df_val_features = feature_engineer.generate_features(
        df_val_raw, split="val", load_cached_data=True
    )

    # 4. Model Training
    print("\n=== Training Models ===")

    # Stage 1: Ridge Regression (OOF)
    # We pass the raw dataframe because Stage 1 needs to access text for TF-IDF internally
    # (or we could pass features if refactored, but StackedRanker expects raw for text access in _get_markdown_data)
    # Actually, looking at modeling.py, train_stage1_ridge_oof takes df_train (raw).
    oof_preds_df = ranker.train_stage1_ridge_oof(df_train_raw, load_cached_data=True)

    # Stage 2: LightGBM
    # Trains on features + OOF preds, validates on val features
    lgbm_model = ranker.train_stage2_lgbm(
        df_train_features, oof_preds_df, df_val_features, df_val_raw
    )

    # 5. Validation & Metric Calculation
    print("\n=== Final Validation Evaluation ===")

    # We need to perform manual inference on validation to get the exact metric format required
    # (Although train_stage2_lgbm prints it, we need to ensure the specific format string)

    # Prepare validation features for prediction
    # We need to merge Ridge predictions first
    ridge_model = joblib.load(Config.RIDGE_MODEL_PATH)

    # Extract text and transform for Ridge
    df_val_md = (
        df_val_raw[df_val_raw["cell_type"] == "markdown"].copy().reset_index(drop=True)
    )
    # Ensure vectorizer is loaded
    ranker.vectorizer.load_models()

    X_val_sparse, _ = ranker.vectorizer.transform(
        df_val_md["source"].astype(str).fillna("").tolist()
    )
    val_ridge_preds = ridge_model.predict(X_val_sparse)

    df_val_ridge = df_val_md[["id", "cell_id"]].copy()
    df_val_ridge["ridge_pred"] = val_ridge_preds

    # Merge into features
    val_data_for_pred = df_val_features.merge(
        df_val_ridge, on=["id", "cell_id"], how="left"
    )

    # Select features for LGBM
    drop_cols = ["id", "cell_id", "rank", "ancestor_id"]
    features = [c for c in val_data_for_pred.columns if c not in drop_cols]

    # Predict Ranks
    val_final_preds = lgbm_model.predict(val_data_for_pred[features])

    # Construct Predictions DataFrame
    df_pred_rows = val_data_for_pred[["id", "cell_id"]].copy()
    df_pred_rows["pred_rank"] = val_final_preds

    # Convert ranks to cell order
    val_code = (
        df_val_raw[df_val_raw["cell_type"] == "code"]
        .groupby("id")["cell_id"]
        .apply(list)
        .to_dict()
    )
    val_md_preds = (
        df_pred_rows.groupby("id")
        .apply(lambda x: dict(zip(x["cell_id"], x["pred_rank"])))
        .to_dict()
    )

    submission_rows = []
    for nb_id in df_val_raw["id"].unique():
        code_cells = val_code.get(nb_id, [])
        md_ranks = val_md_preds.get(nb_id, {})
        ordered_str = convert_ranks_to_order(md_ranks, code_cells)
        submission_rows.append({"id": nb_id, "cell_order": ordered_str})

    df_val_pred_final = pd.DataFrame(submission_rows)
    df_val_gt = df_val_raw[["id", "cell_order"]].drop_duplicates()

    # Compute Metric
    final_metric = kendall_tau_metric(df_val_gt, df_val_pred_final)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error per markdown cell
    # Merge predictions with ground truth ranks
    df_analysis = val_data_for_pred.copy()
    df_analysis["pred_rank"] = val_final_preds

    # Calculate absolute error
    df_analysis["error"] = np.abs(df_analysis["pred_rank"] - df_analysis["rank"])

    # Correlate error with features
    analysis_features = [
        "cell_char_len",
        "cell_word_len",
        "lexical_max_sim",
        "latent_max_sim",
        "notebook_md_count",
        "ridge_pred",
    ]

    print("Correlation between Absolute Error and Features:")
    correlations = (
        df_analysis[analysis_features + ["error"]]
        .corr()["error"]
        .sort_values(ascending=False)
    )
    print(correlations)

    # 7. Submission
    THRESHOLD = 0.7959051868218839

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Load Test Data
        df_test_raw = data_manager.load_data(split="test", load_cached_data=True)

        # Generate Test Features
        df_test_features = feature_engineer.generate_features(
            df_test_raw, split="test", load_cached_data=True
        )

        # Run Inference
        ranker.predict(df_test_raw, df_test_features)

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
