import sys
import os
import shutil
import pandas as pd
import numpy as np

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import NotebookLoader
from library.features import FeaturePipeline
from library.models import Stage2LGBM, SubmissionGenerator
from library.metrics import compute_score


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration for Fast Demonstration
    # --------------------------------------------------------------------------
    print("[Demo] Configuring environment for rapid execution...")

    # Override Config paths to use a demo directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo run to ensure fresh execution
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Initialize directories and seeds
    Config.setup()

    # Override hyperparameters for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 notebooks per split
    Config.MD_VOCAB_SIZE = 1000  # Smaller vocab
    Config.SVD_COMPONENTS = 16  # Fewer components
    Config.NUM_FOLDS = 2  # Fewer folds for Stage 1 OOF
    Config.N_BINS = 5  # Fewer bins for histograms

    # LightGBM speed optimizations
    Config.LGBM_PARAMS["n_estimators"] = 20
    Config.LGBM_PARAMS["min_child_samples"] = 5  # Allow splits on small data
    Config.LGBM_PARAMS["num_leaves"] = 15

    seed_everything()
    logger = get_logger("Demo")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n[Demo] Loading Data...")
    loader = NotebookLoader()

    # Load Train and Validation data
    # force load_cached_data=False to demonstrate raw processing logic
    df_md_train, df_nb_train = loader.load_dataset("train", load_cached_data=False)
    df_md_val, df_nb_val = loader.load_dataset("val", load_cached_data=False)

    # Validation checks
    assert not df_md_train.empty, "Training markdown data is empty."
    assert not df_nb_train.empty, "Training notebook data is empty."
    assert "rank" in df_md_train.columns, "Rank column missing in training data."
    print(
        f"Loaded {len(df_nb_train)} training notebooks and {len(df_md_train)} markdown cells."
    )

    # --------------------------------------------------------------------------
    # 3. Feature Engineering
    # --------------------------------------------------------------------------
    print("\n[Demo] Running Feature Pipeline...")
    pipeline = FeaturePipeline()

    # Fit pipeline (TF-IDF, SVD, Stage 1 Ridge) on training data
    pipeline.fit_pipeline(df_md_train)

    # Transform datasets to get features
    # Note: transform_pipeline handles Stage 1 predictions and Stage 2 feature generation
    df_train_feats = pipeline.transform_pipeline(
        df_md_train, df_nb_train, "train", load_cached_data=False
    )
    df_val_feats = pipeline.transform_pipeline(
        df_md_val, df_nb_val, "val", load_cached_data=False
    )

    # Verify features
    expected_cols = ["pred_ridge", "dist_lex_0", "topk_mean"]
    for col in expected_cols:
        assert col in df_train_feats.columns, f"Feature {col} missing."

    print(f"Generated feature matrix shape: {df_train_feats.shape}")

    # --------------------------------------------------------------------------
    # 4. Model Training (Stage 2)
    # --------------------------------------------------------------------------
    print("\n[Demo] Training Stage 2 LightGBM Model...")
    model = Stage2LGBM()
    model.train(df_train_feats, df_val_feats)

    assert os.path.exists(model.model_path), "LGBM model file was not created."

    # --------------------------------------------------------------------------
    # 5. Validation Scoring
    # --------------------------------------------------------------------------
    print("\n[Demo] Validating Model Performance...")
    # Predict on validation set
    val_preds = model.predict(df_val_feats)
    df_val_feats["pred_rank"] = val_preds

    # Generate validation submission file to calculate metrics
    val_sub_path = os.path.join(Config.WORKING_DIR, "val_predictions.csv")

    # Temporarily point global path to val path
    original_sub_path = Config.SUBMISSION_PATH
    Config.SUBMISSION_PATH = val_sub_path

    # Instantiate generator after updating config so it picks up the new path
    sub_gen = SubmissionGenerator()
    sub_gen.generate(df_val_feats, df_nb_val)

    # Load Ground Truth for Validation
    val_meta_path = os.path.join(Config.METADATA_DIR, "val_metadata.csv")
    df_val_gt = pd.read_csv(val_meta_path)

    # Filter GT to only the notebooks we loaded (due to DEBUG sampling)
    loaded_val_ids = df_nb_val["notebook_id"].unique()
    df_val_gt = df_val_gt[df_val_gt["id"].isin(loaded_val_ids)]

    # Load Predictions
    df_val_pred_file = pd.read_csv(val_sub_path)
    preds_dict = df_val_pred_file.set_index("id")["cell_order"].to_dict()

    # Compute Metric
    score = compute_score(df_val_gt, preds_dict)
    print(f"Validation Kendall Tau Score: {score:.4f}")

    # Basic sanity check on score
    assert -1.0 <= score <= 1.0, f"Score {score} is out of valid range [-1, 1]"

    # --------------------------------------------------------------------------
    # 6. Test Inference & Submission
    # --------------------------------------------------------------------------
    print("\n[Demo] Running Test Inference...")
    Config.SUBMISSION_PATH = original_sub_path  # Restore path

    # Re-instantiate generator to pick up the restored path
    sub_gen = SubmissionGenerator()

    # Load Test Data
    df_md_test, df_nb_test = loader.load_dataset("test", load_cached_data=False)

    if not df_md_test.empty:
        # Generate Features
        df_test_feats = pipeline.transform_pipeline(
            df_md_test, df_nb_test, "test", load_cached_data=False
        )

        # Predict
        test_preds = model.predict(df_test_feats)
        df_test_feats["pred_rank"] = test_preds

        # Generate Submission
        sub_gen.generate(df_test_feats, df_nb_test)

        assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

        # Verify submission format
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        assert list(df_sub.columns) == [
            "id",
            "cell_order",
        ], "Submission columns incorrect."
        assert len(df_sub) == len(df_nb_test), "Submission row count mismatch."

        print(f"Submission generated at {Config.SUBMISSION_PATH}")
        print("Head of submission:")
        print(df_sub.head())
    else:
        print("Test dataset is empty (likely due to sampling). Skipping inference.")

    print("\n[Demo] Completed successfully.")


if __name__ == "__main__":
    main()
