import os
import sys
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, kendall_tau_metric
from library.data_loader import DataLoader
from library.feature_engine import DualViewFeatureGenerator
from library.models import Stage1Ridge, Stage2LGBM


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    Config.setup()
    set_seed(Config.SEED)

    # Fast Baseline Settings
    # We use a subset of data to ensure execution within 2 hours.
    # 10,000 training samples and 2,000 validation samples.
    NUM_TRAIN_SAMPLES = 10000
    NUM_VAL_SAMPLES = 2000

    # Threshold for submission
    SUBMISSION_THRESHOLD = 0.7959051868218839

    print("--- Starting Dual-View Stacked Ranking Pipeline (Fast Baseline) ---")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n[1/7] Loading Data (Debug Mode)...")
    train_md, train_code = DataLoader.load_data(
        "train", load_cached_data=True, debug=True, num_debug_samples=NUM_TRAIN_SAMPLES
    )
    val_md, val_code = DataLoader.load_data(
        "val", load_cached_data=True, debug=True, num_debug_samples=NUM_VAL_SAMPLES
    )

    # --------------------------------------------------------------------------
    # 3. Feature Engineering
    # --------------------------------------------------------------------------
    print("\n[2/7] Generating Features...")
    feature_gen = DualViewFeatureGenerator()

    # Process Training Data
    # Note: This fits the TF-IDF and LSA models
    train_feats = feature_gen.process_data(
        train_md, train_code, "train", load_cached_data=True
    )

    # Process Validation Data
    val_feats = feature_gen.process_data(val_md, val_code, "val", load_cached_data=True)

    # --------------------------------------------------------------------------
    # 4. Stage 1: Ridge Regression (The "Signpost" Model)
    # --------------------------------------------------------------------------
    print("\n[3/7] Training Stage 1 (Ridge)...")
    stage1_model = Stage1Ridge()

    # Ensure vectorizer is available (loaded from cache or memory)
    if feature_gen.tfidf_vectorizer is None:
        train_corpus = train_md["source"].fillna("").astype(str).tolist()
        feature_gen.tfidf_vectorizer = feature_gen._get_tfidf_model(
            corpus=train_corpus, load_cached=True
        )

    # Prepare Sparse Features
    train_text = train_md["source"].fillna("").astype(str).tolist()
    val_text = val_md["source"].fillna("").astype(str).tolist()

    X_train_tfidf = feature_gen.tfidf_vectorizer.transform(train_text)
    X_val_tfidf = feature_gen.tfidf_vectorizer.transform(val_text)
    y_train = train_md["norm_rank"].values

    # Fit and Generate OOF
    stage1_model.fit(X_train_tfidf, y_train)

    train_oof = stage1_model.get_oof_predictions(X_train_tfidf, y_train)
    val_pred_ridge = stage1_model.predict(X_val_tfidf)

    train_feats["ridge_pred"] = train_oof
    val_feats["ridge_pred"] = val_pred_ridge

    # --------------------------------------------------------------------------
    # 5. Stage 2: LightGBM (The "Refinement" Model)
    # --------------------------------------------------------------------------
    print("\n[4/7] Training Stage 2 (LightGBM)...")
    stage2_model = Stage2LGBM()

    # Select Numeric Features
    exclude_cols = ["notebook_id", "cell_id", "source", "rank", "norm_rank"]
    feature_cols = [c for c in train_feats.columns if c not in exclude_cols]
    feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(train_feats[c])
    ]

    print(f"Features used: {feature_cols}")

    stage2_model.fit(
        train_df=train_feats,
        val_df=val_feats,
        feature_cols=feature_cols,
        target_col="norm_rank",
    )

    # --------------------------------------------------------------------------
    # 6. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("\n[5/7] Evaluating on Validation Set...")
    val_final_preds = stage2_model.predict(val_feats, feature_cols)
    val_feats["pred_rank"] = val_final_preds

    # Helper to sort cells
    def post_process_sorting(df_md_pred, df_code):
        submission_rows = []
        md_groups = df_md_pred.groupby("notebook_id")
        code_groups = df_code.groupby("notebook_id")
        all_ids = set(md_groups.groups.keys()) | set(code_groups.groups.keys())

        for nb_id in all_ids:
            cells = []
            n_code = (
                len(code_groups.get_group(nb_id)) if nb_id in code_groups.groups else 0
            )
            n_md = len(md_groups.get_group(nb_id)) if nb_id in md_groups.groups else 0
            total_cells = n_code + n_md

            if total_cells == 0:
                submission_rows.append({"id": nb_id, "cell_order": ""})
                continue

            if n_code > 0:
                c_df = code_groups.get_group(nb_id)
                for _, row in c_df.iterrows():
                    norm_rank = (
                        row["rank"] / (total_cells - 1) if total_cells > 1 else 0.0
                    )
                    cells.append((row["cell_id"], norm_rank))

            if n_md > 0:
                m_df = md_groups.get_group(nb_id)
                for _, row in m_df.iterrows():
                    cells.append((row["cell_id"], row["pred_rank"]))

            cells.sort(key=lambda x: x[1])
            cell_order = " ".join([c[0] for c in cells])
            submission_rows.append({"id": nb_id, "cell_order": cell_order})
        return pd.DataFrame(submission_rows)

    # Reconstruct orders
    val_pred_df = post_process_sorting(val_feats, val_code)

    # Load GT
    df_val_gt = pd.read_csv(Config.VAL_METADATA_PATH)[["id", "cell_order"]]

    # Compute Metric
    score = kendall_tau_metric(val_pred_df, df_val_gt)
    print(f"Final Validation Metric: {score}")

    # --------------------------------------------------------------------------
    # 7. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n[6/7] Performing Failure Analysis...")
    # Calculate error magnitude
    val_feats["abs_error"] = np.abs(val_feats["norm_rank"] - val_feats["pred_rank"])

    # Select features for correlation analysis
    analysis_cols = [
        "abs_error",
        "char_len",
        "word_len",
        "lexical_anchor_sim",
        "semantic_anchor_sim",
        "ridge_pred",
    ]
    # Ensure cols exist
    analysis_cols = [c for c in analysis_cols if c in val_feats.columns]

    corr_matrix = val_feats[analysis_cols].corr()
    error_correlations = corr_matrix["abs_error"].sort_values(ascending=False)

    print("Correlation between Absolute Error and Features:")
    print(error_correlations)

    # --------------------------------------------------------------------------
    # 8. Conditional Submission
    # --------------------------------------------------------------------------
    print("\n[7/7] Checking Submission Criteria...")
    if score > SUBMISSION_THRESHOLD:
        print(
            f"Metric ({score}) > Threshold ({SUBMISSION_THRESHOLD}). Generating Submission..."
        )

        # Load Full Test Data
        # Note: We use full test data here, not debug, as submission requires all IDs
        test_md, test_code = DataLoader.load_data(
            "test", load_cached_data=True, debug=False
        )

        # Generate Test Features
        test_feats = feature_gen.process_data(
            test_md, test_code, "test", load_cached_data=True
        )

        # Stage 1 Inference
        test_text = test_md["source"].fillna("").astype(str).tolist()
        X_test_tfidf = feature_gen.tfidf_vectorizer.transform(test_text)
        test_feats["ridge_pred"] = stage1_model.predict(X_test_tfidf)

        # Stage 2 Inference
        test_final_preds = stage2_model.predict(test_feats, feature_cols)
        test_feats["pred_rank"] = test_final_preds

        # Sort and Save
        submission_df = post_process_sorting(test_feats, test_code)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Metric ({score}) <= Threshold ({SUBMISSION_THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
