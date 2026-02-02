import sys
import os
import pandas as pd
import numpy as np

# Import library modules
import library.config as config
import library.preprocessor as preprocessor
import library.feature_engineering as fe
import library.models as models
from library.data_loader import NotebookLoader
from library.preprocessor import fit_transform_corpus, transform_cells
from library.feature_engineering import NeighborhoodFeatureExtractor
from library.models import StackedHybridRanker
from library.postprocessing import OrderReconstructor
from library.metrics import compute_score


def setup_fast_execution():
    """
    Overrides configuration constants in imported modules to ensure
    the demo runs quickly (within seconds/minutes) instead of hours.
    """
    print(">>> Setting up fast execution parameters...")

    # 1. Set Seed
    config.set_seed(42)

    # 2. Override Preprocessor Configs (imported as values in preprocessor.py)
    preprocessor.VOCAB_SIZE = 1000
    preprocessor.SVD_COMPONENTS = 10

    # 3. Override Model Configs
    # LGBM_PARAMS is a dictionary, so modifying it here reflects everywhere
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["num_leaves"] = 8

    # NUM_FOLDS is an integer imported in models.py
    models.NUM_FOLDS = 2

    # 4. Use a separate cache directory for this demo to avoid conflicts
    demo_cache = os.path.join(config.WORKING_DIR, "demo_run")
    config.CACHE_DIR = demo_cache
    preprocessor.CACHE_DIR = demo_cache
    fe.CACHE_DIR = demo_cache
    models.CACHE_DIR = demo_cache

    os.makedirs(demo_cache, exist_ok=True)


def main():
    # Apply speed optimizations
    setup_fast_execution()

    # =========================================================================
    # 1. Data Loading
    # =========================================================================
    print("\n>>> 1. Loading Data (Debug Mode)")
    loader = NotebookLoader()

    # debug=True loads a small sample (100 notebooks)
    df_train, df_val, df_test = loader.get_partitioned_data(
        load_cached_data=False, debug=True
    )

    print(f"Train DataFrame shape: {df_train.shape}")
    print(f"Val DataFrame shape:   {df_val.shape}")

    # Validation
    assert not df_train.empty, "Training data should not be empty"
    assert not df_val.empty, "Validation data should not be empty"
    assert "source" in df_train.columns
    assert "cell_type" in df_train.columns

    # =========================================================================
    # 2. Preprocessing (Text Vectorization)
    # =========================================================================
    print("\n>>> 2. Training Vectorization Pipeline")
    # Fits TF-IDF and SVD on the training markdown cells
    pipeline = fit_transform_corpus(df_train, load_cached_models=False)

    # Validate Pipeline
    assert pipeline.is_fitted, "Pipeline should be fitted"

    print("Transforming cell text to matrices...")
    train_tfidf, train_svd = transform_cells(df_train, pipeline)
    val_tfidf, val_svd = transform_cells(df_val, pipeline)

    # Validate Dimensions (SVD_COMPONENTS was set to 10)
    assert train_svd.shape[1] == 10, f"Expected SVD dim 10, got {train_svd.shape[1]}"
    assert val_svd.shape[0] == len(df_val), "Validation matrix rows mismatch"

    # =========================================================================
    # 3. Feature Engineering (Neighborhood)
    # =========================================================================
    print("\n>>> 3. Extracting Neighborhood Features")
    extractor = NeighborhoodFeatureExtractor()

    # Extract features based on similarity between markdown and code cells
    train_neighbors = extractor.extract_features(
        df_train, train_tfidf, train_svd, "demo_train", load_cached_data=False
    )
    val_neighbors = extractor.extract_features(
        df_val, val_tfidf, val_svd, "demo_val", load_cached_data=False
    )

    # Validate Feature Extraction
    assert not train_neighbors.empty, "Train neighborhood features empty"
    # Check for expected columns (e.g., lex_sim_0, lat_mean_rank)
    expected_col = "lex_sim_0"
    assert expected_col in train_neighbors.columns, f"Missing feature {expected_col}"

    # =========================================================================
    # 4. Modeling - Stage 1: Ridge Regression (OOF)
    # =========================================================================
    print("\n>>> 4. Modeling Stage 1: Ridge Regression")
    modeler = StackedHybridRanker()

    # Train Ridge on Train set (generating OOF predictions via CV)
    train_oof = modeler.train_stage1_ridge(df_train, pipeline, load_cached_data=False)

    # Predict on Validation set using the fitted Ridge model
    val_ridge_preds = modeler.predict_stage1_ridge(df_val, pipeline)

    # Validate Stage 1 Outputs
    assert "ridge_pred" in train_oof.columns
    assert "ridge_pred" in val_ridge_preds.columns
    assert len(val_ridge_preds) == len(df_val[df_val["cell_type"] == "markdown"])

    # =========================================================================
    # 5. Modeling - Stage 2: LightGBM (Refinement)
    # =========================================================================
    print("\n>>> 5. Modeling Stage 2: LightGBM")

    # Construct final feature sets (Metadata + Neighbors + Ridge Preds)
    train_stage2 = extractor.construct_stage2_features(
        train_neighbors, train_oof, df_train, "demo_train", load_cached_data=False
    )
    val_stage2 = extractor.construct_stage2_features(
        val_neighbors, val_ridge_preds, df_val, "demo_val", load_cached_data=False
    )

    # Train LightGBM
    modeler.train_stage2_lgbm(train_stage2, val_stage2, load_cached_data=False)

    # Predict on Validation set
    val_final_preds = modeler.predict_stage2_lgbm(val_stage2)

    # Validate Stage 2 Outputs
    assert "pred_rank" in val_final_preds.columns
    assert not val_final_preds.isnull().values.any(), "Predictions contain NaNs"

    # =========================================================================
    # 6. Post-processing & Evaluation
    # =========================================================================
    print("\n>>> 6. Post-processing and Evaluation")
    reconstructor = OrderReconstructor()

    # Reconstruct the full cell order (Code + Markdown) for validation notebooks
    val_predictions_dict = reconstructor.reconstruct_order(df_val, val_final_preds)

    # Prepare Ground Truth for Metric Calculation
    # We reconstruct the ground truth strings from the loaded validation dataframe
    gt_rows = []
    for nb_id, group in df_val.groupby("notebook_id"):
        # Sort by the ground truth 'rank' column
        sorted_group = group.sort_values("rank")
        order_str = " ".join(sorted_group["cell_id"].values)
        gt_rows.append({"id": nb_id, "cell_order": order_str})

    df_val_gt = pd.DataFrame(gt_rows)

    # Compute Kendall Tau Score
    score = compute_score(df_val_gt, val_predictions_dict)

    print(f"\nFinal Validation Kendall Tau Score: {score:.4f}")

    # Final Sanity Check
    assert -1.0 <= score <= 1.0, "Score is mathematically impossible"

    print("\n>>> Demo Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
