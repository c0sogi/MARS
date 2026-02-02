import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.pipeline import HybridRankingPipeline


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("Initializing Runfile...")
    # Ensure reproducibility
    Config.set_seed(Config.RANDOM_SEED)

    # Define working directory for this run
    # The Config class already sets this to "./working/idea_14"
    print(f"Working Directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Pipeline Training
    # --------------------------------------------------------------------------
    pipeline = HybridRankingPipeline()

    # Train the pipeline
    # This handles data loading, feature generation, model training, and metric computation
    print("\n" + "=" * 40)
    print(" STARTING PIPELINE TRAINING ")
    print("=" * 40)

    # We use load_cached_data=True to leverage any existing preprocessed files
    # to speed up the run if restarted.
    val_metric = pipeline.train(load_cached_data=True)

    # REQUIRED: Print Validation Metric in specific format
    print(f"Final Validation Metric: {val_metric}")

    # --------------------------------------------------------------------------
    # 3. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" FAILURE ANALYSIS ")
    print("=" * 40)

    try:
        # Load cached validation data and features generated during training
        val_processed_path = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
        val_features_path = os.path.join(Config.WORKING_DIR, "features_val.parquet")

        if os.path.exists(val_processed_path) and os.path.exists(val_features_path):
            df_val = pd.read_parquet(val_processed_path)
            feats_val = pd.read_parquet(val_features_path)

            # Filter df_val to markdown only to match features
            df_val_md = df_val[df_val["cell_type"] == "markdown"].reset_index(drop=True)

            # Ensure alignment (merge on notebook_id and cell_id)
            # feats_val already contains the targets 'pct_rank'
            analysis_df = pd.merge(
                feats_val,
                df_val_md[["notebook_id", "cell_id", "source_clean"]],
                on=["notebook_id", "cell_id"],
                how="inner",
            )

            # Calculate text length features
            analysis_df["char_len"] = analysis_df["source_clean"].fillna("").apply(len)

            # Re-run Inference to get predictions
            # 1. TF-IDF Transform
            print("Generating validation predictions for analysis...")
            pipeline.feature_gen.pipeline.load_models()
            tfidf_matrix, _ = pipeline.feature_gen.pipeline.transform(
                analysis_df["source_clean"].fillna("").tolist()
            )

            # 2. Stage 1 Predict
            pipeline.stage1.load()
            ridge_preds = pipeline.stage1.predict(tfidf_matrix)

            # 3. Stage 2 Predict
            # Prepare stack: Ridge Preds + Anchor Features
            # Note: _prepare_stacking_features expects the full features dataframe
            X_stack = pipeline._prepare_stacking_features(ridge_preds, analysis_df)

            pipeline.stage2.load()
            lgbm_preds = pipeline.stage2.predict(X_stack)

            # Calculate Error
            analysis_df["pred_rank"] = lgbm_preds
            analysis_df["error"] = np.abs(
                analysis_df["pct_rank"] - analysis_df["pred_rank"]
            )

            # Compute Correlations
            correlations = {}
            features_to_check = [
                "lexical_anchor",
                "latent_anchor",
                "symbolic_anchor",
                "char_len",
                "pct_rank",
            ]

            print("\nCorrelation between Absolute Error and Features:")
            for feat in features_to_check:
                if feat in analysis_df.columns:
                    corr, _ = pearsonr(analysis_df["error"], analysis_df[feat])
                    correlations[feat] = corr
                    print(f"  {feat}: {corr:.4f}")

            # Identify worst performing notebooks
            nb_error = (
                analysis_df.groupby("notebook_id")["error"]
                .mean()
                .sort_values(ascending=False)
            )
            print(f"\nTop 3 Notebooks with highest error:\n{nb_error.head(3)}")

        else:
            print(
                "Cached validation files not found. Skipping detailed failure analysis."
            )

    except Exception as e:
        print(f"An error occurred during failure analysis: {e}")

    # --------------------------------------------------------------------------
    # 4. Submission
    # --------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" SUBMISSION GENERATION ")
    print("=" * 40)

    THRESHOLD = 0.7959051868218839

    if val_metric > THRESHOLD:
        print(
            f"Validation metric ({val_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        pipeline.predict(load_cached_data=True)
    else:
        print(
            f"Validation metric ({val_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
