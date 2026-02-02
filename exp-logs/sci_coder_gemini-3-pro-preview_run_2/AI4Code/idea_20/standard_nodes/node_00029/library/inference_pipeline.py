import os
import numpy as np
import pandas as pd
from library.config import Config
from library.data_loader import load_notebooks, load_metadata
from library.vectorizer import TextVectorizer
from library.feature_extractor import AnchorFeatureGenerator
from library.model_wrapper import Stage1Ridge, Stage2LGBM
from library.utils import format_submission


def run_inference(debug: bool = False, load_cached_data: bool = True):
    """
    Executes the inference pipeline for the test set.

    Steps:
    1. Load test data and models.
    2. Generate Stage 1 predictions (Ridge).
    3. Extract anchor features and merge Stage 1 preds.
    4. Generate Stage 2 predictions (LightGBM).
    5. Sort cells (Code + Markdown) by rank.
    6. Generate submission file.

    Args:
        debug (bool): If True, runs on a subset of data (defined in Config).
        load_cached_data (bool): If True, attempts to load intermediate files from cache.
    """
    # Initialize configuration
    Config.setup()

    print(f"Starting Inference Pipeline (Debug={debug}, Cache={load_cached_data})")

    # --------------------------------------------------------------------------
    # 1. Load Test Data
    # --------------------------------------------------------------------------
    print("\n[Step 1/7] Loading Test Notebooks...")
    test_df = load_notebooks("test", load_cached_data=load_cached_data, debug=debug)

    # --------------------------------------------------------------------------
    # 2. Load Vectorizer
    # --------------------------------------------------------------------------
    print("\n[Step 2/7] Loading Text Vectorizer...")
    vectorizer = TextVectorizer()
    vec_base_path = os.path.join(Config.WORKING_DIR, "text_vectorizer")

    if os.path.exists(f"{vec_base_path}_tfidf.joblib"):
        vectorizer.load(vec_base_path)
    else:
        raise FileNotFoundError(
            f"TextVectorizer not found at {vec_base_path}. Train the model first."
        )

    # --------------------------------------------------------------------------
    # 3. Stage 1 Prediction (Ridge)
    # --------------------------------------------------------------------------
    print("\n[Step 3/7] Stage 1 - Ridge Prediction...")
    ridge_model = Stage1Ridge()
    ridge_path = os.path.join(Config.WORKING_DIR, "stage1_ridge_model")

    if os.path.exists(f"{ridge_path}.joblib"):
        ridge_model.load(ridge_path)
    else:
        raise FileNotFoundError(
            f"Stage 1 model not found at {ridge_path}. Train the model first."
        )

    # Filter Markdown cells for prediction
    test_md = test_df[test_df["cell_type"] == "markdown"].reset_index(drop=True)

    if not test_md.empty:
        print("Transforming test text to sparse TF-IDF...")
        # Fill NaNs to ensure robustness
        X_test_sparse = vectorizer.transform(test_md["source"].fillna("").astype(str))

        print("Predicting with Ridge model...")
        s1_preds = ridge_model.predict(X_test_sparse)
        test_md["stage1_pred"] = s1_preds
    else:
        print("Warning: No markdown cells found in test set.")
        test_md["stage1_pred"] = []

    # --------------------------------------------------------------------------
    # 4. Feature Extraction
    # --------------------------------------------------------------------------
    print("\n[Step 4/7] Extracting Anchor Features...")
    feature_gen = AnchorFeatureGenerator(vectorizer)
    test_features = feature_gen.extract_features(
        test_df, "test", load_cached_data=load_cached_data
    )

    # --------------------------------------------------------------------------
    # 5. Prepare Stage 2 Dataset
    # --------------------------------------------------------------------------
    print("\n[Step 5/7] Preparing Stage 2 Dataset...")
    # Merge Stage 1 predictions into the feature DataFrame
    test_final = test_features.merge(
        test_md[["id", "cell_id", "stage1_pred"]], on=["id", "cell_id"], how="left"
    )

    # Define feature columns (exclude metadata)
    exclude_cols = [
        "id",
        "cell_id",
        "norm_rank",
        "cell_type",
        "source",
        "ancestor_id",
        "parent_id",
    ]
    feature_cols = [c for c in test_final.columns if c not in exclude_cols]

    # --------------------------------------------------------------------------
    # 6. Stage 2 Prediction (LightGBM)
    # --------------------------------------------------------------------------
    print("\n[Step 6/7] Stage 2 - LightGBM Prediction...")
    lgbm_model = Stage2LGBM()
    lgbm_path = os.path.join(Config.WORKING_DIR, "stage2_lgbm")

    if os.path.exists(f"{lgbm_path}.txt"):
        lgbm_model.load(lgbm_path)
    else:
        raise FileNotFoundError(
            f"Stage 2 model not found at {lgbm_path}. Train the model first."
        )

    if not test_final.empty:
        s2_preds = lgbm_model.predict(test_final, feature_cols)
        test_final["pred_rank"] = s2_preds
    else:
        test_final["pred_rank"] = []

    # --------------------------------------------------------------------------
    # 7. Sort and Format Submission
    # --------------------------------------------------------------------------
    print("\n[Step 7/7] Generating Submission...")

    # 7a. Assign Ranks to Code Cells
    # We assume code cells in test_df are in correct relative order (as read from JSON)
    code_cells = test_df[test_df["cell_type"] == "code"].copy()

    if not code_cells.empty:
        # Generate equidistant ranks [0, 1] for code cells within each notebook
        # GroupBy transform ensures we process per notebook ID
        code_cells["rank"] = code_cells.groupby("id")["cell_id"].transform(
            lambda x: np.linspace(0, 1, len(x))
        )
    else:
        code_cells["rank"] = []

    # 7b. Prepare Markdown Cells with Predicted Ranks
    md_cells = test_final[["id", "cell_id", "pred_rank"]].rename(
        columns={"pred_rank": "rank"}
    )

    # 7c. Combine and Sort
    all_cells = pd.concat(
        [code_cells[["id", "cell_id", "rank"]], md_cells[["id", "cell_id", "rank"]]],
        ignore_index=True,
    )

    # Sort by ID first, then by Rank
    all_cells = all_cells.sort_values(["id", "rank"])

    # 7d. Aggregate into List
    submission_series = all_cells.groupby("id")["cell_id"].apply(list)

    # 7e. Ensure all Test IDs are present
    # Load test metadata to get the authoritative list of IDs
    test_meta = load_metadata("test")
    required_ids = test_meta["id"].unique()

    # Convert series to DataFrame
    sub_df = submission_series.reset_index()
    sub_df.columns = ["id", "cell_order"]

    # Merge with required IDs to handle notebooks that might have been lost (e.g. empty)
    final_df = pd.DataFrame({"id": required_ids})
    final_df = final_df.merge(sub_df, on="id", how="left")

    # Prepare lists for format_submission
    final_ids = final_df["id"].tolist()
    final_orders = []

    for val in final_df["cell_order"]:
        if isinstance(val, list):
            final_orders.append(val)
        else:
            # Handle missing/empty notebooks
            final_orders.append([])

    # Save Submission
    out_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    format_submission(final_ids, final_orders, out_path)

    print(f"Submission saved to {out_path}")
