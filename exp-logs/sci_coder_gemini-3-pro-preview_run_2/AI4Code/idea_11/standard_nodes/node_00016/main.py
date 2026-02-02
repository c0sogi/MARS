import os
import sys
import pandas as pd
import numpy as np
from library import (
    config,
    utils,
    feature_engine,
    model_zoo,
    training_pipeline,
    inference_pipeline,
)


def main():
    # 1. Setup
    utils.set_seed()
    utils.log_message("Starting runfile.py execution...")

    # 2. Load/Generate Features for Training
    # This ensures TF-IDF and SVD are fit on Train data
    utils.log_message("Generating/Loading Training Features...")
    df_train_feats = feature_engine.generate_features(
        mode="train", load_cached_data=True
    )

    # 3. Run Stage 1 (Ridge) Cross-Validation
    # This returns OOF predictions for markdown cells in df_train and trains the full Ridge model
    utils.log_message("Running Stage 1 (Ridge) CV...")
    oof_preds = training_pipeline.run_stage1_cv(df_train_feats, load_cached_data=True)

    # 4. Load/Generate Features for Validation
    utils.log_message("Generating/Loading Validation Features...")
    df_val_feats = feature_engine.generate_features(mode="val", load_cached_data=True)

    # 5. Prepare Stage 2 (LightGBM) Data
    utils.log_message("Preparing Stage 2 Data...")

    # --- Prepare Training Data ---
    mask_train_md = df_train_feats["is_code"] == 0
    df_train_md = df_train_feats[mask_train_md].copy()

    # Define feature columns (Must match library logic)
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
    svd_cols = [c for c in df_train_feats.columns if c.startswith("svd_")]
    feature_cols.extend(svd_cols)

    X_train = df_train_md[feature_cols].copy()
    X_train["ridge_pred"] = oof_preds
    y_train = df_train_md["rank"].values

    # --- Prepare Validation Data ---
    mask_val_md = df_val_feats["is_code"] == 0
    df_val_md = df_val_feats[mask_val_md].copy()

    # Generate Ridge Predictions for Validation
    utils.log_message("Generating Stage 1 Predictions for Validation...")
    ridge_model = model_zoo.Stage1Ridge().load()

    # Load Vectorizer and Transform Val Text
    vectorizer = feature_engine.TextVectorizer()
    vectorizer.load(config.CACHE_TFIDF_VECTORIZER)
    X_val_tfidf = vectorizer.transform(df_val_md["source"].astype(str).tolist())

    val_ridge_preds = ridge_model.predict(X_val_tfidf)

    X_val = df_val_md[feature_cols].copy()
    X_val["ridge_pred"] = val_ridge_preds
    y_val = df_val_md["rank"].values

    # 6. Train Stage 2 (LightGBM)
    utils.log_message("Training Stage 2 (LightGBM)...")
    lgbm = model_zoo.Stage2LGBM()
    lgbm.fit(X_train, y_train, X_val, y_val)
    lgbm.save()

    # 7. Validation Evaluation
    utils.log_message("Evaluating Model...")
    val_preds = lgbm.predict(X_val)

    # Reconstruct Order and Compute Kendall Tau
    # We implement the reconstruction logic here to ensure we capture the score correctly
    df_val_eval = df_val_feats.copy()

    # Initialize pred_rank with NaN
    df_val_eval["pred_rank"] = np.nan
    # Fill predictions for markdown cells
    df_val_eval.loc[mask_val_md, "pred_rank"] = val_preds

    predicted_orders = []

    # Group by ID to reconstruct order
    for nb_id, group in df_val_eval.groupby("id", observed=True):
        code_cells = group[group["is_code"] == 1].copy()
        md_cells = group[group["is_code"] == 0].copy()

        n_code = len(code_cells)

        # Code cells are anchors at fixed positions: 0, 1, 2...
        # Normalize to [0, 1] range for sorting against predicted MD ranks
        if n_code > 0:
            # Code rank in data_factory is the integer index
            code_cells["sort_rank"] = code_cells["rank"] / n_code
        else:
            code_cells["sort_rank"] = code_cells["rank"]  # Fallback

        md_cells["sort_rank"] = md_cells["pred_rank"]

        combined = pd.concat([code_cells, md_cells])
        combined = combined.sort_values("sort_rank")

        cell_order_str = " ".join(combined["cell_id"].astype(str).tolist())
        predicted_orders.append({"id": nb_id, "cell_order": cell_order_str})

    df_pred_orders = pd.DataFrame(predicted_orders)

    # Load Ground Truth
    df_true_orders = pd.read_csv(config.VAL_METADATA_PATH)[["id", "cell_order"]]

    final_metric = utils.kendall_tau_metric(df_true_orders, df_pred_orders)
    print(f"Final Validation Metric: {final_metric}")

    # 8. Failure Analysis
    utils.log_message("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_val - val_preds)

    # Create a dataframe for correlation analysis
    analysis_df = X_val.copy()
    analysis_df["error"] = errors

    # Compute correlations
    correlations = analysis_df.corrwith(analysis_df["error"]).sort_values(
        ascending=False
    )

    print("Correlation between Error Magnitude and Features:")
    print(correlations.head(10))

    # 9. Submission
    THRESHOLD = 0.7959051868218839
    if final_metric > THRESHOLD:
        utils.log_message(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        inference_pipeline.run_inference_pipeline(load_cached_data=True)
    else:
        utils.log_message(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
