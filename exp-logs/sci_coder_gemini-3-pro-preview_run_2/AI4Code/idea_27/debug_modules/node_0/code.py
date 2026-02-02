import os
import shutil
import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.metrics import mean_absolute_error

# Import from provided library files
from library.config import Config
from library.utils import set_seed, compute_kendall_tau
from library.data_manager import NotebookLoader
from library.vectorization import TextProcessor
from library.feature_extraction import FeatureEngineer
from library.models import Stage1Ridge, Stage2LGBM

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Initializing Demonstration...")

    # --------------------------------------------------------------------------
    # 1. Setup Configuration and Environment
    # --------------------------------------------------------------------------
    # Create a subclass of Config to override settings for a fast demo run
    class DemoConfig(Config):
        # Paths
        WORKING_DIR = "./working/demo_run"
        METADATA_DIR = "./working/demo_metadata"
        SUBMISSION_PATH = "./working/demo_submission/submission.csv"

        # Reduced Hyperparameters for Speed
        TFIDF_MAX_FEATURES = 1000  # Reduced from 60000
        SVD_COMPONENTS = 16  # Reduced from 128
        N_FOLDS = 2  # Reduced from 5

        # Reduced LGBM complexity
        LGBM_PARAMS = {
            "objective": "mae",
            "metric": "mae",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "n_estimators": 50,  # Reduced from 3000
            "early_stopping_rounds": 10,
            "verbose": -1,
            "n_jobs": 1,
            "random_state": 42,
        }

    config = DemoConfig()
    set_seed(42)

    # Clean up previous demo runs if they exist
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    if os.path.exists(config.METADATA_DIR):
        shutil.rmtree(config.METADATA_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.METADATA_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Prepare Subset Metadata (Optimization for Speed)
    # --------------------------------------------------------------------------
    print("Creating subset metadata for rapid execution...")

    # Load original metadata
    orig_train_meta = pd.read_csv("./metadata/train_metadata.csv")
    orig_val_meta = pd.read_csv("./metadata/val_metadata.csv")
    orig_test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # Sample 50 notebooks for training, 20 for validation, 20 for testing
    # We ensure we don't sample more than available
    n_train = min(50, len(orig_train_meta))
    n_val = min(20, len(orig_val_meta))
    n_test = min(20, len(orig_test_meta))

    subset_train = orig_train_meta.sample(n=n_train, random_state=42)
    subset_val = orig_val_meta.sample(n=n_val, random_state=42)
    subset_test = orig_test_meta.sample(n=n_test, random_state=42)

    # Save to the demo metadata directory
    subset_train.to_csv(
        os.path.join(config.METADATA_DIR, "train_metadata.csv"), index=False
    )
    subset_val.to_csv(
        os.path.join(config.METADATA_DIR, "val_metadata.csv"), index=False
    )
    subset_test.to_csv(
        os.path.join(config.METADATA_DIR, "test_metadata.csv"), index=False
    )

    print(
        f"Subset created: Train={len(subset_train)}, Val={len(subset_val)}, Test={len(subset_test)}"
    )

    # --------------------------------------------------------------------------
    # 3. Data Loading
    # --------------------------------------------------------------------------
    print("Loading data using NotebookLoader...")
    loader = NotebookLoader(config)

    # prepare_datasets loads train and val based on the metadata files we just created
    # We set load_cached_data=False to force processing of our new subset
    df_train_corpus, df_val_corpus = loader.prepare_datasets(load_cached_data=False)

    # Verify data loading
    assert len(df_train_corpus) > 0, "Train corpus is empty"
    assert len(df_val_corpus) > 0, "Val corpus is empty"
    assert "rank" in df_train_corpus.columns, "Rank column missing in train corpus"
    print(
        f"Loaded {len(df_train_corpus)} training cells and {len(df_val_corpus)} validation cells."
    )

    # Combine for full training context (simulating the pipeline flow)
    df_full_corpus = pd.concat([df_train_corpus, df_val_corpus], ignore_index=True)

    # --------------------------------------------------------------------------
    # 4. Vectorization (TF-IDF & SVD)
    # --------------------------------------------------------------------------
    print("Running TextProcessor (TF-IDF + SVD)...")
    processor = TextProcessor(config)

    # Fit on the corpus
    processor.fit_pipeline(df_full_corpus, load_cached_models=False)

    # Transform
    tfidf_mat, svd_mat = processor.transform_cells(
        df_full_corpus, mode="train", load_cached_data=False
    )

    # Verify shapes
    assert tfidf_mat.shape[0] == len(
        df_full_corpus
    ), "TF-IDF matrix rows mismatch corpus size"
    assert svd_mat.shape[0] == len(
        df_full_corpus
    ), "SVD matrix rows mismatch corpus size"
    assert svd_mat.shape[1] == config.SVD_COMPONENTS, "SVD components mismatch config"
    print(
        f"Vectorization complete. TF-IDF shape: {tfidf_mat.shape}, SVD shape: {svd_mat.shape}"
    )

    # --------------------------------------------------------------------------
    # 5. Feature Extraction
    # --------------------------------------------------------------------------
    print("Running FeatureEngineer...")
    engineer = FeatureEngineer(config)

    df_features = engineer.extract_features(
        df_full_corpus, tfidf_mat, svd_mat, mode="train", load_cached_data=False
    )

    # Verify features
    # We expect LSA features and Neighborhood features
    expected_cols = [f"lsa_{i}" for i in range(min(16, config.SVD_COMPONENTS))]
    expected_cols += ["lex_top1_sim", "lat_mean_rank", "md_ratio"]

    for col in expected_cols:
        assert col in df_features.columns, f"Expected feature {col} missing"

    # Verify target exists
    assert "target" in df_features.columns, "Target column missing in features"
    print(f"Feature extraction complete. Feature matrix shape: {df_features.shape}")

    # --------------------------------------------------------------------------
    # 6. Model Training - Stage 1 (Ridge)
    # --------------------------------------------------------------------------
    print("Training Stage 1 (Ridge Regression)...")
    stage1 = Stage1Ridge(config)

    # Fit OOF
    oof_preds = stage1.fit_oof(
        df_full_corpus, tfidf_mat, df_features, load_cached_data=False
    )

    # Verify predictions
    assert len(oof_preds) == len(df_features), "OOF predictions length mismatch"
    assert (
        oof_preds.min() >= 0 and oof_preds.max() <= 1
    ), "OOF predictions out of range [0, 1]"

    # Attach OOF to features for Stage 2
    df_features["ridge_pred"] = oof_preds

    # Calculate MAE for Stage 1
    mae_stage1 = mean_absolute_error(df_features["target"], oof_preds)
    print(f"Stage 1 Ridge MAE: {mae_stage1:.4f}")

    # --------------------------------------------------------------------------
    # 7. Model Training - Stage 2 (LightGBM)
    # --------------------------------------------------------------------------
    print("Training Stage 2 (LightGBM)...")
    stage2 = Stage2LGBM(config)

    stage2.fit(df_features, load_cached_data=False)

    # Verify model file creation
    assert os.path.exists(stage2.model_path), "Stage 2 model file not saved"
    print("Stage 2 training complete.")

    # --------------------------------------------------------------------------
    # 8. Inference on Test Set
    # --------------------------------------------------------------------------
    print("Running Inference on Test Subset...")

    # Load test data
    df_test_corpus = loader.load_test_data(load_cached_data=False)

    # Transform test data
    tfidf_test, svd_test = processor.transform_cells(
        df_test_corpus, mode="test", load_cached_data=False
    )

    # Extract features
    df_test_features = engineer.extract_features(
        df_test_corpus, tfidf_test, svd_test, mode="test", load_cached_data=False
    )

    # Stage 1 Prediction
    ridge_test_preds = stage1.predict(df_test_corpus, tfidf_test, df_test_features)
    df_test_features["ridge_pred"] = ridge_test_preds

    # Stage 2 Prediction
    final_preds = stage2.predict(df_test_features)
    df_test_features["pred_rank"] = final_preds

    print(f"Inference complete. Generated {len(final_preds)} predictions.")

    # --------------------------------------------------------------------------
    # 9. Post-Processing & Submission Generation
    # --------------------------------------------------------------------------
    print("Generating Submission File...")

    submission_rows = []
    pred_map = dict(zip(df_test_features["cell_id"], df_test_features["pred_rank"]))

    for nb_id, group in df_test_corpus.groupby("id"):
        cells = group.copy()

        # Code cells get fixed ranks
        code_mask = cells["cell_type"] == "code"
        n_code = code_mask.sum()
        if n_code > 0:
            cells.loc[code_mask, "rank"] = np.linspace(0, 1, n_code)

        # Markdown cells get predicted ranks
        md_mask = cells["cell_type"] == "markdown"
        cells.loc[md_mask, "rank"] = cells.loc[md_mask, "cell_id"].map(pred_map)

        # Handle any NaNs (e.g., if a markdown cell was somehow missed, though unlikely)
        cells["rank"] = cells["rank"].fillna(0.5)

        # Sort
        cells = cells.sort_values("rank")
        cell_order = " ".join(cells["cell_id"].tolist())

        submission_rows.append({"id": nb_id, "cell_order": cell_order})

    df_submission = pd.DataFrame(submission_rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    df_submission.to_csv(config.SUBMISSION_PATH, index=False)

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(df_submission.head())

    # --------------------------------------------------------------------------
    # 10. Validation Metric Calculation (Kendall Tau)
    # --------------------------------------------------------------------------
    print("Calculating Kendall Tau on Validation Subset...")

    # To demonstrate the metric utility, we'll run inference on the validation set
    # (In a real scenario, this would be done via cross-validation, but here we just
    # want to show the utility function works).

    # We already have df_features for validation (it was part of df_full_corpus)
    # Let's filter for validation IDs
    val_ids = subset_val["id"].unique()
    val_features = df_features[df_features["id"].isin(val_ids)].copy()

    if len(val_features) > 0:
        # Predict using Stage 2 model
        val_preds = stage2.predict(val_features)
        val_features["pred_rank"] = val_preds

        # Reconstruct order for validation notebooks
        val_pred_rows = []
        val_pred_map = dict(zip(val_features["cell_id"], val_features["pred_rank"]))

        # We need the full validation corpus to get code cells
        val_corpus_subset = df_val_corpus[df_val_corpus["id"].isin(val_ids)]

        for nb_id, group in val_corpus_subset.groupby("id"):
            cells = group.copy()
            code_mask = cells["cell_type"] == "code"
            n_code = code_mask.sum()
            if n_code > 0:
                cells.loc[code_mask, "rank"] = np.linspace(0, 1, n_code)

            md_mask = cells["cell_type"] == "markdown"
            cells.loc[md_mask, "rank"] = cells.loc[md_mask, "cell_id"].map(val_pred_map)
            cells["rank"] = cells["rank"].fillna(0.5)

            cells = cells.sort_values("rank")
            cell_order = " ".join(cells["cell_id"].tolist())
            val_pred_rows.append({"id": nb_id, "cell_order": cell_order})

        df_val_preds = pd.DataFrame(val_pred_rows)

        # Compute Metric
        score = compute_kendall_tau(subset_val, df_val_preds)
        print(f"Validation Kendall Tau Score: {score:.4f}")
    else:
        print("No validation features found (sample size might be too small).")

    print("\nDemonstration Completed Successfully.")


if __name__ == "__main__":
    main()
