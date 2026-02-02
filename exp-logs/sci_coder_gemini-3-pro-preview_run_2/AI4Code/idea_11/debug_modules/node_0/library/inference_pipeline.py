import os
import pandas as pd
import numpy as np
from library import config, utils, feature_engine, model_zoo


def generate_predictions(load_cached_data=True):
    """
    Generates predictions for the test set using the trained 2-stage pipeline.

    Args:
        load_cached_data (bool): Whether to use cached features if available.

    Returns:
        pd.DataFrame: Test DataFrame containing all cells with an added 'pred_rank'
                      column for markdown cells.
    """
    utils.log_message("Generating features for test set...")
    # 1. Generate/Load Features for Test
    # This handles loading raw data, vectorizing, SVD, and neighborhood extraction
    df_test = feature_engine.generate_features(
        mode="test", load_cached_data=load_cached_data
    )

    # 2. Prepare Data for Inference
    # Filter for Markdown cells (the ones we need to rank)
    mask_md = df_test["is_code"] == 0
    df_md = df_test[mask_md].copy()

    if len(df_md) == 0:
        utils.log_message("Warning: No markdown cells found in test set.")
        df_test["pred_rank"] = np.nan
        return df_test

    # 3. Stage 1 Inference (Ridge)
    utils.log_message("Running Stage 1 (Ridge) Inference...")

    # Load Vectorizer
    vectorizer = feature_engine.TextVectorizer()
    if not os.path.exists(config.CACHE_TFIDF_VECTORIZER):
        raise FileNotFoundError("TF-IDF Vectorizer not found. Train the model first.")
    vectorizer.load(config.CACHE_TFIDF_VECTORIZER)

    # Transform text
    X_tfidf = vectorizer.transform(df_md["source"].astype(str).tolist())

    # Load Ridge Model
    ridge_model = model_zoo.Stage1Ridge().load()

    # Predict
    ridge_preds = ridge_model.predict(X_tfidf)

    # 4. Stage 2 Inference (LightGBM)
    utils.log_message("Running Stage 2 (LightGBM) Inference...")

    # Construct Feature Matrix
    # Must match the columns used in training_pipeline.py
    feature_cols = [
        "lex_mean_rank",
        "lex_weighted_rank",
        "lex_max_sim",
        "lat_mean_rank",
        "lat_weighted_rank",
        "lat_max_sim",
        "n_code_cells",
        "md_ratio",
    ]
    # Add SVD columns
    svd_cols = [c for c in df_test.columns if c.startswith("svd_")]
    feature_cols.extend(svd_cols)

    # Prepare X_test
    X_test_lgbm = df_md[feature_cols].copy()
    X_test_lgbm["ridge_pred"] = ridge_preds

    # Load LightGBM Model
    lgbm_model = model_zoo.Stage2LGBM().load()

    # Predict
    final_preds = lgbm_model.predict(X_test_lgbm)

    # 5. Merge Predictions
    # Initialize pred_rank with NaN
    df_test["pred_rank"] = np.nan
    # Assign predictions to markdown cells
    df_test.loc[mask_md, "pred_rank"] = final_preds

    return df_test


def postprocess_ordering(df_test):
    """
    Converts predicted ranks into the final cell order string for submission.
    Merges fixed code cell positions with predicted markdown positions.

    Args:
        df_test (pd.DataFrame): Test DataFrame with 'pred_rank', 'is_code', 'n_code_cells'.

    Returns:
        pd.DataFrame: Submission DataFrame with columns ['id', 'cell_order'].
    """
    utils.log_message("Post-processing cell orders...")

    submission_rows = []

    # Group by notebook ID
    # Use observed=True to handle categorical 'id' efficiently
    grouped = df_test.groupby("id", observed=True)

    for nb_id, group in grouped:
        # Get number of code cells (constant for the group)
        # Handle case where column might be NaN if something failed, though unlikely
        n_code = group["n_code_cells"].iloc[0]

        # Separate Code and Markdown
        code_cells = group[group["is_code"] == 1].copy()
        md_cells = group[group["is_code"] == 0].copy()

        # Assign Ranks to Code Cells
        # Code cells are anchors: 0, 1, 2...
        # We normalize them to be comparable with predicted MD ranks [0, 1]
        # Rank = Index / Total_Code_Cells
        if n_code > 0:
            # The code cells in the DataFrame are in their original correct order
            # (preserved from JSON list order by data_factory)
            code_ranks = np.arange(len(code_cells)) / n_code
            code_cells["final_rank"] = code_ranks
        else:
            # If no code cells, rank is irrelevant for code, but MD ranks are used directly
            code_cells["final_rank"] = []  # Empty

        # Assign Ranks to Markdown Cells
        # Use the predictions from the model
        md_cells["final_rank"] = md_cells["pred_rank"]

        # Combine and Sort
        combined = pd.concat([code_cells, md_cells])
        combined = combined.sort_values("final_rank")

        # Extract ID sequence
        cell_order = combined["cell_id"].astype(str).tolist()
        cell_order_str = " ".join(cell_order)

        submission_rows.append({"id": nb_id, "cell_order": cell_order_str})

    submission_df = pd.DataFrame(submission_rows)
    return submission_df


def run_inference_pipeline(load_cached_data=True):
    """
    Main function to run the inference pipeline and generate submission.csv.
    """
    utils.set_seed()
    utils.log_message("Starting Inference Pipeline...")

    # 1. Generate Predictions
    df_test_preds = generate_predictions(load_cached_data=load_cached_data)

    # 2. Post-process to get final ordering
    submission_df = postprocess_ordering(df_test_preds)

    # 3. Save Submission
    utils.log_message(f"Saving submission to {config.SUBMISSION_PATH}...")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)

    utils.log_message("Inference Pipeline Completed Successfully.")
