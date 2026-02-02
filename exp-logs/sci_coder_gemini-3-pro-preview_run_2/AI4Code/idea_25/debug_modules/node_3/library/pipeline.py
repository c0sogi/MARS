import os
import pandas as pd
import numpy as np
from library.config import WORKING_DIR, SUBMISSION_DIR, RANDOM_STATE
from library.utils import seed_everything, kendall_tau_metric
from library.data_loader import load_data
from library.text_processing import get_vectorizer
from library.feature_extraction import NeighborhoodExtractor, assemble_stage2_features
from library.models import Stage1Ridge, Stage2LGBM


def _get_code_ranks(df_cells):
    """
    Assigns fixed equidistant ranks (0.0 to 1.0) to code cells per notebook.
    Assumes code cells in df_cells are in the correct relative order.
    """
    df_code = df_cells[df_cells["cell_type"] == "code"].copy()

    # We need to assign ranks 0..1 based on the sequence of code cells
    # Since data_loader preserves input order, we rely on that.

    # Helper to calculate linspace ranks
    def get_linspace_ranks(group):
        n = len(group)
        if n == 0:
            return []
        if n == 1:
            return [0.0]
        return np.linspace(0.0, 1.0, n)

    # Apply per notebook
    # We use a trick to be fast: groupby -> transform is slow for custom funcs
    # Instead, we just count and compute mathematically

    # 1. Enumerate code cells within each notebook
    df_code["code_idx"] = df_code.groupby("notebook_id").cumcount()

    # 2. Get total code cells per notebook
    code_counts = df_code.groupby("notebook_id")["cell_id"].transform("count")

    # 3. Compute rank: idx / (count - 1)
    # Handle division by zero (count=1) -> rank=0.0
    df_code["rank"] = df_code["code_idx"] / (code_counts - 1)
    df_code.loc[code_counts <= 1, "rank"] = 0.0

    return df_code[["cell_id", "notebook_id", "rank"]]


def _postprocess_submission(df_md_preds, df_cells):
    """
    Combines predicted markdown ranks with fixed code ranks to form cell_order.
    """
    # 1. Get Code Ranks (Anchors)
    df_code_ranks = _get_code_ranks(df_cells)

    # 2. Prepare Markdown Ranks
    # df_md_preds has ['cell_id', 'final_rank']
    # We need to ensure we have notebook_id
    if "notebook_id" not in df_md_preds.columns:
        # Merge to get notebook_id if missing
        df_md_preds = pd.merge(
            df_md_preds, df_cells[["cell_id", "notebook_id"]], on="cell_id", how="left"
        )

    df_md_ranks = df_md_preds[["cell_id", "notebook_id", "final_rank"]].rename(
        columns={"final_rank": "rank"}
    )

    # 3. Concatenate
    df_full = pd.concat([df_code_ranks, df_md_ranks], axis=0, ignore_index=True)

    # 4. Sort
    df_full = df_full.sort_values(by=["notebook_id", "rank"])

    # 5. Group and join
    submission = (
        df_full.groupby("notebook_id")["cell_id"]
        .apply(" ".join)
        .reset_index()
        .rename(columns={"cell_id": "cell_order"})
    )

    return submission


def train_pipeline(debug_n=None, load_cached_data=True):
    """
    Executes the training pipeline:
    1. Load Data
    2. Vectorize
    3. Stage 1 Ridge (Train + OOF + Val Preds)
    4. Feature Extraction (Neighborhoods)
    5. Stage 2 LGBM (Train)
    6. Validation Scoring
    """
    seed_everything(RANDOM_STATE)
    print("=== Starting Training Pipeline ===")

    # 1. Load Data
    print("Loading data...")
    df_train = load_data("train", load_cached_data=load_cached_data, debug_n=debug_n)
    df_val = load_data("val", load_cached_data=load_cached_data, debug_n=debug_n)

    # 2. Vectorization
    print("Initializing Vectorizer...")
    # We fit on train source.
    train_source = df_train["source"].astype(str).fillna("")
    val_source = df_val["source"].astype(str).fillna("")

    vectorizer = get_vectorizer(train_texts=train_source, load_cached=load_cached_data)

    # Transform for Stage 1 (Ridge only needs TF-IDF)
    print("Transforming text for Stage 1...")
    X_train_tfidf, _ = vectorizer.transform(train_source)
    X_val_tfidf, _ = vectorizer.transform(val_source)

    # Targets for Stage 1 (Ridge predicts rank directly? Or pct_rank?)
    # We predict pct_rank.
    y_train = df_train["pct_rank"].values

    # 3. Stage 1: Ridge
    ridge_model = Stage1Ridge()

    # Fit Ridge (for final artifact)
    ridge_model.fit(X_train_tfidf, y_train)

    # Generate OOF for Stage 2 Training
    # Only for Markdown cells?
    # Usually we train on all cells or just MD?
    # The prompt implies predicting MD order.
    # However, Ridge is trained on all cells usually to learn the structure.
    # But Stage 2 is specifically for MD cells.
    # We will generate OOF for ALL cells in train, then filter for MD in assembly.
    df_oof_ridge = ridge_model.get_oof_predictions(
        X_train_tfidf,
        y_train,
        df_train["cell_id"].values,
        load_cached_data=load_cached_data,
    )

    # Generate Val Predictions
    val_preds_ridge = ridge_model.predict(X_val_tfidf)
    df_val_ridge = pd.DataFrame(
        {"cell_id": df_val["cell_id"].values, "ridge_rank": val_preds_ridge}
    )

    # 4. Feature Extraction
    extractor = NeighborhoodExtractor()

    df_train_neigh = extractor.extract_neighborhood_features(
        df_train, vectorizer, split="train", load_cached_data=load_cached_data
    )

    df_val_neigh = extractor.extract_neighborhood_features(
        df_val, vectorizer, split="val", load_cached_data=load_cached_data
    )

    # 5. Stage 2: LGBM
    print("Assembling Stage 2 Features...")
    # Train set
    df_train_s2 = assemble_stage2_features(df_train_neigh, df_oof_ridge, df_train)
    # Target for Stage 2: We need to map cell_id to pct_rank
    # df_train_s2 has cell_id. We merge target.
    df_train_s2 = pd.merge(
        df_train_s2, df_train[["cell_id", "pct_rank"]], on="cell_id", how="inner"
    )

    # Val set
    df_val_s2 = assemble_stage2_features(df_val_neigh, df_val_ridge, df_val)
    df_val_s2 = pd.merge(
        df_val_s2, df_val[["cell_id", "pct_rank"]], on="cell_id", how="inner"
    )

    # Prepare X, y
    drop_cols = ["cell_id", "notebook_id", "pct_rank"]
    features = [c for c in df_train_s2.columns if c not in drop_cols]

    X_train_lgbm = df_train_s2[features]
    y_train_lgbm = df_train_s2["pct_rank"]

    X_val_lgbm = df_val_s2[features]
    y_val_lgbm = df_val_s2["pct_rank"]

    lgbm_model = Stage2LGBM()
    lgbm_model.fit(X_train_lgbm, y_train_lgbm, X_val=X_val_lgbm, y_val=y_val_lgbm)

    # 6. Validation Scoring
    print("Running Validation Scoring...")
    # Predict on Val
    val_final_preds = lgbm_model.predict(X_val_lgbm)

    df_val_preds_md = pd.DataFrame(
        {"cell_id": df_val_s2["cell_id"], "final_rank": val_final_preds}
    )

    # Reconstruct Order
    # We need the original df_val to get code cells
    df_pred_order = _postprocess_submission(df_val_preds_md, df_val)

    # Construct GT Order from df_val (which is loaded in correct order)
    df_gt_order = (
        df_val.groupby("notebook_id")["cell_id"]
        .apply(" ".join)
        .reset_index()
        .rename(columns={"cell_id": "cell_order"})
    )

    score = kendall_tau_metric(df_pred_order, df_gt_order)
    print(f"Validation Kendall Tau: {score}")

    return score


def inference_pipeline(debug_n=None, load_cached_data=True):
    """
    Executes the inference pipeline:
    1. Load Test Data
    2. Load Models
    3. Vectorize & Ridge Predict
    4. Feature Extraction
    5. LGBM Predict
    6. Post-process & Submit
    """
    seed_everything(RANDOM_STATE)
    print("=== Starting Inference Pipeline ===")

    # 1. Load Data
    print("Loading test data...")
    df_test = load_data("test", load_cached_data=load_cached_data, debug_n=debug_n)

    # 2. Load Models
    print("Loading models...")
    vectorizer = get_vectorizer(load_cached=True)

    ridge_model = Stage1Ridge()
    if not ridge_model.load_model():
        raise RuntimeError("Stage 1 Ridge model not found. Run training first.")

    lgbm_model = Stage2LGBM()
    if not lgbm_model.load_model():
        raise RuntimeError("Stage 2 LGBM model not found. Run training first.")

    # 3. Vectorize & Ridge
    print("Generating Stage 1 Predictions...")
    test_source = df_test["source"].astype(str).fillna("")
    X_test_tfidf, _ = vectorizer.transform(test_source)

    test_ridge_preds = ridge_model.predict(X_test_tfidf)
    df_test_ridge = pd.DataFrame(
        {"cell_id": df_test["cell_id"].values, "ridge_rank": test_ridge_preds}
    )

    # 4. Feature Extraction
    # Note: load_cached_data=False typically for inference to ensure fresh run on test set,
    # unless we want to resume a crashed run.
    df_test_neigh = NeighborhoodExtractor().extract_neighborhood_features(
        df_test, vectorizer, split="test", load_cached_data=load_cached_data
    )

    # 5. LGBM Predict
    print("Generating Stage 2 Predictions...")
    df_test_s2 = assemble_stage2_features(df_test_neigh, df_test_ridge, df_test)

    drop_cols = ["cell_id", "notebook_id", "pct_rank"]  # pct_rank won't exist in test
    features = [c for c in df_test_s2.columns if c not in drop_cols]

    # Ensure feature order matches training (simple check, ideally we'd save feature list)
    # Assuming deterministic column generation in assemble_stage2_features
    X_test_lgbm = df_test_s2[features]

    test_final_preds = lgbm_model.predict(X_test_lgbm)

    df_test_preds_md = pd.DataFrame(
        {"cell_id": df_test_s2["cell_id"], "final_rank": test_final_preds}
    )

    # 6. Post-process
    print("Post-processing and Saving Submission...")
    submission = _postprocess_submission(df_test_preds_md, df_test)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
