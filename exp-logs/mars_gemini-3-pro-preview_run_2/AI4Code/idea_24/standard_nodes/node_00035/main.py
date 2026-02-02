import os
import pandas as pd
import numpy as np
import torch
import warnings
from scipy.stats import pearsonr

# Import from provided library files
from library.config import set_seed, METADATA_DIR, SUBMISSION_DIR, CACHE_DIR
from library.data_loader import NotebookLoader
from library.preprocessor import fit_transform_corpus, transform_cells
from library.feature_engineering import NeighborhoodFeatureExtractor
from library.models import StackedHybridRanker
from library.postprocessing import OrderReconstructor
from library.metrics import compute_score, save_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_pipeline():
    print("=================================================================")
    print(" STARTING PIPELINE EXECUTION")
    print("=================================================================")

    # 1. Setup
    set_seed()

    # Define sample sizes to ensure execution within time limits
    # The feature extraction loop in Python is the bottleneck.
    # 20k train notebooks is sufficient for a strong baseline.
    TRAIN_SAMPLE = 20000
    VAL_SAMPLE = 5000

    # 2. Data Loading
    print("\n[Step 1/7] Loading Data...")
    loader = NotebookLoader()

    # Load Training Data
    df_train = loader.load_notebooks(
        os.path.join(METADATA_DIR, "train_metadata.csv"),
        "train",
        load_cached_data=True,
        sample_size=TRAIN_SAMPLE,
    )

    # Load Validation Data
    df_val = loader.load_notebooks(
        os.path.join(METADATA_DIR, "val_metadata.csv"),
        "val",
        load_cached_data=True,
        sample_size=VAL_SAMPLE,
    )

    # Load Test Data (Full)
    df_test = loader.load_notebooks(
        os.path.join(METADATA_DIR, "test_metadata.csv"),
        "test",
        load_cached_data=True,
        sample_size=None,
    )

    print(
        f"Data Loaded: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)} cells"
    )

    # 3. Vectorization
    print("\n[Step 2/7] Vectorizing Text...")
    # Fit pipeline on training data
    pipeline = fit_transform_corpus(df_train, load_cached_models=True)

    # Transform all partitions
    print("Transforming Training Data...")
    train_tfidf, train_svd = transform_cells(df_train, pipeline)
    print("Transforming Validation Data...")
    val_tfidf, val_svd = transform_cells(df_val, pipeline)
    print("Transforming Test Data...")
    test_tfidf, test_svd = transform_cells(df_test, pipeline)

    # 4. Stage 1: Ridge Regression
    print("\n[Step 3/7] Stage 1: Ridge Regression...")
    ranker = StackedHybridRanker()

    # Train Ridge and get OOF preds for training set
    train_oof_preds = ranker.train_stage1_ridge(
        df_train, pipeline, load_cached_data=True
    )

    # Predict on Validation and Test
    val_ridge_preds = ranker.predict_stage1_ridge(df_val, pipeline)
    test_ridge_preds = ranker.predict_stage1_ridge(df_test, pipeline)

    # 5. Feature Engineering
    print("\n[Step 4/7] Feature Engineering (Multi-Resolution Neighborhoods)...")
    extractor = NeighborhoodFeatureExtractor()

    # Extract raw neighborhood features
    # This step involves loops over notebooks, so we rely on the sample sizes set earlier
    train_feats = extractor.extract_features(
        df_train, train_tfidf, train_svd, "train", load_cached_data=True
    )
    val_feats = extractor.extract_features(
        df_val, val_tfidf, val_svd, "val", load_cached_data=True
    )
    test_feats = extractor.extract_features(
        df_test, test_tfidf, test_svd, "test", load_cached_data=True
    )

    # Construct final Stage 2 feature matrices
    train_s2 = extractor.construct_stage2_features(
        train_feats, train_oof_preds, df_train, "train", load_cached_data=True
    )
    val_s2 = extractor.construct_stage2_features(
        val_feats, val_ridge_preds, df_val, "val", load_cached_data=True
    )
    test_s2 = extractor.construct_stage2_features(
        test_feats, test_ridge_preds, df_test, "test", load_cached_data=True
    )

    # 6. Stage 2: LightGBM
    print("\n[Step 5/7] Stage 2: LightGBM Training...")
    ranker.train_stage2_lgbm(train_s2, val_s2, load_cached_data=True)

    # Generate Predictions
    print("Generating predictions...")
    val_final_preds = ranker.predict_stage2_lgbm(val_s2)
    test_final_preds = ranker.predict_stage2_lgbm(test_s2)

    # 7. Validation & Analysis
    print("\n[Step 6/7] Validation & Analysis...")
    reconstructor = OrderReconstructor()

    # Reconstruct Validation Orders
    val_predictions = reconstruct_order(df_val, val_final_preds, reconstructor)

    # Load Ground Truth for Validation
    # We need to load the metadata file again to get the 'cell_order' string for the sampled notebooks
    df_val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    # Filter to only the notebooks we processed
    processed_val_ids = df_val["notebook_id"].unique()
    df_val_gt = df_val_meta[df_val_meta["id"].isin(processed_val_ids)].copy()

    # Compute Metric
    print("Computing Kendall Tau...")
    score = compute_score(df_val_gt, val_predictions)
    print(f"Final Validation Metric: {score}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Merge predictions with targets and features to analyze errors
    # val_s2 contains 'rank' (target) and features. val_final_preds contains 'pred_rank'.
    # Note: val_s2 ranks are normalized (0-1) but stored as 'rank' column in raw form in df_val?
    # Wait, construct_stage2_features keeps 'rank' from df_train/val which is integer rank.
    # LightGBM training converts it to normalized inside train_stage2_lgbm.
    # Let's reconstruct the normalized target for analysis.

    analysis_df = val_s2.merge(val_final_preds, on="cell_id", how="inner")

    # Calculate normalized target
    denoms = analysis_df["total_cells"].values - 1
    denoms = np.maximum(denoms, 1)
    analysis_df["target_norm"] = analysis_df["rank"] / denoms

    # Calculate Error
    analysis_df["error"] = np.abs(analysis_df["pred_rank"] - analysis_df["target_norm"])

    # Correlate Error with Features
    features_to_check = [
        "md_ratio",
        "total_cells",
        "ridge_pred",
        "lex_sim_0",
        "lat_sim_0",
        "lex_mean_rank",
        "lat_mean_rank",
    ]

    print("Correlation of Absolute Error with Features:")
    for feat in features_to_check:
        if feat in analysis_df.columns:
            corr, _ = pearsonr(analysis_df["error"], analysis_df[feat])
            print(f"  {feat}: {corr:.4f}")

    # 8. Submission
    print("\n[Step 7/7] Submission Generation...")
    THRESHOLD = 0.7959051868218839

    if score > THRESHOLD:
        print(
            f"Validation score ({score:.6f}) exceeds threshold ({THRESHOLD:.6f}). Generating submission."
        )

        # Reconstruct Test Orders
        test_predictions = reconstruct_order(df_test, test_final_preds, reconstructor)

        # Save
        save_submission(test_predictions, "submission.csv")
    else:
        print(
            f"Validation score ({score:.6f}) does not exceed threshold ({THRESHOLD:.6f}). Skipping submission."
        )

    print("\nPipeline execution complete.")


def reconstruct_order(df_cells, preds_df, reconstructor):
    """
    Helper to bridge the gap between raw cell dataframes and the reconstructor.
    """
    # The reconstructor expects the test_df structure (code cells present)
    # and a predictions dataframe.
    return reconstructor.reconstruct_order(df_cells, preds_df)


if __name__ == "__main__":
    run_pipeline()
