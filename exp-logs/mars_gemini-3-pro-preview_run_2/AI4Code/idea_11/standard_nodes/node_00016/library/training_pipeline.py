import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from library import config, utils, feature_engine, model_zoo


def _evaluate_kendall_tau(df_val, preds, mode="val"):
    """
    Internal helper to evaluate Kendall Tau score on validation data.
    Reconstructs cell order from predicted ranks and compares with ground truth.
    """
    # Create a working copy
    df_eval = df_val.copy()

    # preds is an array of ranks for markdown cells only
    # We need to map these back to the dataframe
    # Assuming df_eval is the full validation set (code + md)

    # Assign predictions to markdown cells
    df_eval.loc[df_eval["is_code"] == 0, "pred_rank"] = preds

    # For code cells, assign their fixed rank (which is effectively their index in code sequence)
    # In data_factory, 'rank' for code cells is the integer index.
    # We need to scale this to be comparable with predicted MD ranks (0.0 to 1.0)
    # However, the sorting strategy is: sort by rank.
    # MD ranks are predicted in [0, 1].
    # Code ranks in data_factory are 0.0, 1.0, 2.0...
    # We need to normalize code ranks for sorting: rank / num_code_cells

    # Calculate num_code_cells per notebook if not present or rely on data_factory logic
    # data_factory: cell_data["rank"] = float(code_rank_map.get(cid, -1)) for code

    # Let's re-calculate normalized code ranks for sorting purposes
    # We can use the 'n_code_cells' feature if available, or compute on fly.
    # To be robust, we compute on fly.

    # We need to process per notebook
    predicted_orders = []

    # Group by ID
    for nb_id, group in df_eval.groupby("id", observed=True):
        # Separate code and md
        code_cells = group[group["is_code"] == 1].copy()
        md_cells = group[group["is_code"] == 0].copy()

        n_code = len(code_cells)

        # Normalize code ranks: 0..N-1 -> 0..1 (roughly)
        # We place code cell i at position i.
        # Ideally, we want code cell i to be at rank i / n_code (approx)
        # The target generation used: rank = skeleton_pos / num_code_cells
        # So code cell i is effectively at rank i / n_code.
        if n_code > 0:
            code_cells["sort_rank"] = code_cells["rank"] / n_code
        else:
            code_cells["sort_rank"] = code_cells[
                "rank"
            ]  # Should be empty or irrelevant

        md_cells["sort_rank"] = md_cells["pred_rank"]

        # Combine and sort
        combined = pd.concat([code_cells, md_cells])
        combined = combined.sort_values("sort_rank")

        # Extract order
        cell_order_str = " ".join(combined["cell_id"].tolist())
        predicted_orders.append({"id": nb_id, "cell_order": cell_order_str})

    df_pred = pd.DataFrame(predicted_orders)

    # Ground Truth
    # We need the ground truth cell orders.
    # We can get this from the metadata file or reconstruct from df_val if we trust it.
    # To be safe, we read the metadata file directly.
    if mode == "val":
        meta_path = config.VAL_METADATA_PATH
    else:
        meta_path = config.TRAIN_METADATA_PATH

    df_true = pd.read_csv(meta_path)[["id", "cell_order"]]

    score = utils.kendall_tau_metric(df_true, df_pred)
    return score


def run_stage1_cv(df_train, load_cached_data=True):
    """
    Runs 5-Fold Cross-Validation for Stage 1 (Ridge) to generate OOF predictions.
    Also trains the Ridge model on the full dataset.
    """
    utils.log_message("\n=== Running Stage 1: Ridge Regression CV ===")

    # Check cache for OOF predictions
    if load_cached_data and os.path.exists(config.CACHE_STAGE1_OOF):
        utils.log_message(
            f"Loading cached OOF predictions from {config.CACHE_STAGE1_OOF}..."
        )
        try:
            oof_df = pd.read_parquet(config.CACHE_STAGE1_OOF)
            # We need to ensure these align with the df_train markdown cells
            # We will merge or align by index.
            # Returning the series aligned to df_train[is_code==0]
            # Assuming cache was saved correctly with index or id/cell_id
            return oof_df["ridge_pred"].values
        except Exception as e:
            utils.log_message(f"Failed to load OOF cache: {e}. Recomputing...")

    # Filter for Markdown cells
    mask_md = df_train["is_code"] == 0
    df_md = df_train[mask_md].copy()

    # Load Vectorizer (should have been fit by feature_engine)
    vectorizer = feature_engine.TextVectorizer()
    if not os.path.exists(config.CACHE_TFIDF_VECTORIZER):
        raise FileNotFoundError(
            "TF-IDF Vectorizer not found. Run feature generation first."
        )
    vectorizer.load(config.CACHE_TFIDF_VECTORIZER)

    # Transform text
    utils.log_message("Vectorizing training text for Ridge CV...")
    X_tfidf = vectorizer.transform(df_md["source"].astype(str).tolist())
    y = df_md["rank"].values
    groups = df_md["ancestor_id"].values

    # Initialize OOF array
    oof_preds = np.zeros(len(df_md))

    # 5-Fold Group CV
    gkf = GroupKFold(n_splits=5)

    model = model_zoo.Stage1Ridge()

    utils.log_message("Starting Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_tfidf, y, groups)):
        X_fold_train = X_tfidf[train_idx]
        y_fold_train = y[train_idx]
        X_fold_val = X_tfidf[val_idx]

        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_val)

        oof_preds[val_idx] = preds
        utils.log_message(f"Fold {fold+1} completed.")

    # Save OOF predictions
    # We save as a dataframe with ID keys to be safe, or just the aligned column if we trust index
    # To be safe for the pipeline, we save the aligned values with the index of df_md
    oof_df = pd.DataFrame({"ridge_pred": oof_preds}, index=df_md.index)
    utils.log_message(f"Saving OOF predictions to {config.CACHE_STAGE1_OOF}...")
    oof_df.to_parquet(config.CACHE_STAGE1_OOF)

    # Train on Full Dataset
    utils.log_message("Retraining Ridge on full training set...")
    model.fit(X_tfidf, y)
    model.save()

    return oof_preds


def train_stacking_model(df_train, oof_preds, df_val, load_cached_data=True):
    """
    Trains the Stage 2 (LightGBM) model using OOF predictions and neighborhood features.
    """
    utils.log_message("\n=== Running Stage 2: LightGBM Stacking ===")

    # 1. Prepare Training Data
    utils.log_message("Preparing Stage 2 Training Data...")
    mask_train_md = df_train["is_code"] == 0
    df_train_md = df_train[mask_train_md].copy()

    # Features list
    # Neighborhood + Metadata + SVD
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
    svd_cols = [c for c in df_train.columns if c.startswith("svd_")]
    feature_cols.extend(svd_cols)

    # Construct X_train
    X_train = df_train_md[feature_cols].copy()
    X_train["ridge_pred"] = oof_preds
    y_train = df_train_md["rank"].values

    # 2. Prepare Validation Data
    utils.log_message("Preparing Stage 2 Validation Data...")
    mask_val_md = df_val["is_code"] == 0
    df_val_md = df_val[mask_val_md].copy()

    # Generate Ridge Predictions for Validation
    # Load Ridge Model
    ridge_model = model_zoo.Stage1Ridge().load()

    # Vectorize Val Text
    vectorizer = feature_engine.TextVectorizer()
    vectorizer.load(config.CACHE_TFIDF_VECTORIZER)
    X_val_tfidf = vectorizer.transform(df_val_md["source"].astype(str).tolist())

    # Predict
    val_ridge_preds = ridge_model.predict(X_val_tfidf)

    # Construct X_val
    X_val = df_val_md[feature_cols].copy()
    X_val["ridge_pred"] = val_ridge_preds
    y_val = df_val_md["rank"].values

    # 3. Train LightGBM
    lgbm = model_zoo.Stage2LGBM()
    lgbm.fit(X_train, y_train, X_val, y_val)
    lgbm.save()

    # 4. Final Evaluation
    utils.log_message("Evaluating Final Model on Validation Set...")
    final_val_preds = lgbm.predict(X_val)

    kendall_score = _evaluate_kendall_tau(df_val, final_val_preds, mode="val")
    utils.log_message(f"Final Validation Kendall Tau: {kendall_score}")

    return lgbm


def run_training_pipeline(load_cached_data=True):
    """
    Orchestrates the full training pipeline.
    """
    utils.set_seed()
    utils.log_message("Starting Training Pipeline...")

    # 1. Generate/Load Features for Train
    # This ensures TF-IDF and SVD are fit on Train
    df_train_feats = feature_engine.generate_features(
        mode="train", load_cached_data=load_cached_data
    )

    # 2. Run Stage 1 CV (Ridge)
    # Returns OOF predictions for markdown cells in df_train
    oof_preds = run_stage1_cv(df_train_feats, load_cached_data=load_cached_data)

    # 3. Generate/Load Features for Val
    df_val_feats = feature_engine.generate_features(
        mode="val", load_cached_data=load_cached_data
    )

    # 4. Train Stage 2 (LightGBM)
    train_stacking_model(
        df_train_feats, oof_preds, df_val_feats, load_cached_data=load_cached_data
    )

    utils.log_message("Training Pipeline Completed Successfully.")
