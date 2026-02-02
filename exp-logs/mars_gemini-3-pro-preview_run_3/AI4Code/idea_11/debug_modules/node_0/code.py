import os
import pandas as pd
import numpy as np
import torch
import shutil
import warnings
import logging

# Import from provided library files
from library.config import Config, set_seed
from library.utils import read_notebook, preprocess_text, kendall_tau, format_submission
from library.data_loader import (
    NotebookTextLoader,
    prepare_relaxed_pairs,
    FineTuningDataset,
)
from library.backbone import BackboneTrainer
from library.feature_extractor import FeatureEngineer
from library.ranker import LGBMRanker
from library.inference import InferencePipeline

# Suppress warnings and logs for clean output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_demo_environment():
    """
    Sets up a temporary working directory and creates small subsets of the metadata
    to allow the pipeline to run quickly for demonstration purposes.
    """
    print(">>> Setting up demo environment...")

    # Define demo directory
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths and parameters for speed
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "fine_tuned_mpnet")
    Config.LGBM_MODEL_PATH = os.path.join(demo_dir, "lgbm_model.txt")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Cache paths
    Config.TRAIN_PAIRS_PATH = os.path.join(demo_dir, "train_pairs_relaxed.parquet")
    Config.TRAIN_FEATURES_PATH = os.path.join(demo_dir, "train_features.parquet")
    Config.VAL_FEATURES_PATH = os.path.join(demo_dir, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(demo_dir, "test_features_debug.parquet")

    # Reduce computational parameters
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.FT_SAMPLE_SIZE = 50  # Only use 50 notebooks for fine-tuning
    Config.LGBM_PARAMS["n_estimators"] = 10  # Very few trees for demo
    Config.LGBM_PARAMS["verbose"] = -1

    # Create subset metadata files
    # We read the original metadata and sample 20 rows for train, val, and test
    # This ensures the feature extractor and other components run almost instantly.

    # 1. Train Subset
    df_train = pd.read_csv("./metadata/train.csv")
    df_train_sub = df_train.head(20).copy()
    demo_train_path = os.path.join(demo_dir, "train.csv")
    df_train_sub.to_csv(demo_train_path, index=False)
    Config.TRAIN_PATH = demo_train_path

    # 2. Val Subset
    df_val = pd.read_csv("./metadata/val.csv")
    df_val_sub = df_val.head(20).copy()
    demo_val_path = os.path.join(demo_dir, "val.csv")
    df_val_sub.to_csv(demo_val_path, index=False)
    Config.VAL_PATH = demo_val_path

    # 3. Test Subset
    df_test = pd.read_csv("./metadata/test.csv")
    df_test_sub = df_test.head(20).copy()
    demo_test_path = os.path.join(demo_dir, "test.csv")
    df_test_sub.to_csv(demo_test_path, index=False)
    Config.TEST_PATH = demo_test_path

    print(f"Demo environment configured at {demo_dir}")
    print("Config parameters updated for fast execution.")


def test_utils():
    """
    Demonstrates and verifies utility functions.
    """
    print("\n>>> Testing Utils...")

    # 1. Test read_notebook
    # Pick a file from the demo train set
    df_train = pd.read_csv(Config.TRAIN_PATH)
    sample_row = df_train.iloc[0]
    file_path = os.path.join(Config.INPUT_ROOT, sample_row["file_path"])

    code_cells, md_cells = read_notebook(file_path)

    print(
        f"Read notebook {sample_row['id']}: {len(code_cells)} code cells, {len(md_cells)} markdown cells."
    )

    # Assertions
    assert isinstance(code_cells, list), "code_cells should be a list"
    assert isinstance(md_cells, list), "md_cells should be a list"
    if len(code_cells) > 0:
        assert isinstance(
            code_cells[0], tuple
        ), "Cell items should be tuples (id, source)"

    # 2. Test preprocess_text
    raw_text = "  Import NumPy as NP  "
    clean_text = preprocess_text(raw_text)
    assert clean_text == "import numpy as np", "Text preprocessing failed"

    # 3. Test kendall_tau
    # Perfect match
    gt = [["a", "b", "c"]]
    pred = [["a", "b", "c"]]
    score = kendall_tau(gt, pred)
    assert abs(score - 1.0) < 1e-6, f"Expected 1.0 for perfect match, got {score}"

    # Complete inversion (n=2) -> 1 swap. Max swaps = n(n-1)/2 = 1. Score = 1 - 4(1/1) = -3?
    # Wait, formula is 1 - 4 * (S / Total_Max_S_over_dataset).
    # For a single notebook n=2: Max Swaps = 2*1 = 2 (Wait, the metric definition says sum n(n-1)).
    # Metric definition: K = 1 - 4 * Sum(S) / Sum(n*(n-1))
    # n=2. n(n-1) = 2.
    # Inversion: [a, b] vs [b, a]. Swaps=1.
    # K = 1 - 4 * (1 / 2) = 1 - 2 = -1.
    gt_inv = [["a", "b"]]
    pred_inv = [["b", "a"]]
    score_inv = kendall_tau(gt_inv, pred_inv)
    assert (
        abs(score_inv - (-1.0)) < 1e-6
    ), f"Expected -1.0 for inversion, got {score_inv}"

    print("Utils verification passed.")


def test_backbone_fine_tuning():
    """
    Demonstrates fine-tuning the sentence transformer backbone.
    """
    print("\n>>> Testing Backbone Fine-Tuning...")

    trainer = BackboneTrainer()

    # Train using the demo data (Config.TRAIN_PATH is already patched)
    # This will generate 'train_pairs_relaxed.parquet' in the demo dir
    trainer.train(load_cached_data=False)

    # Verify model was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Fine-tuned model directory not created."
    assert os.path.exists(
        os.path.join(Config.MODEL_SAVE_PATH, "config.json")
    ), "Model config not found."

    # Test encoding with the new model
    embeddings = trainer.encode(["test code", "test markdown"], batch_size=2)
    assert embeddings.shape == (
        2,
        768,
    ), f"Expected embedding shape (2, 768), got {embeddings.shape}"

    print("Backbone fine-tuning verification passed.")
    return trainer


def test_feature_extraction(backbone_trainer):
    """
    Demonstrates feature extraction using the fine-tuned backbone.
    """
    print("\n>>> Testing Feature Extraction...")

    engineer = FeatureEngineer(backbone_model=backbone_trainer)

    # Extract features for Train (Demo subset)
    # We force load_cached_data=False to ensure logic runs
    train_features = engineer.extract_features(
        metadata_path=Config.TRAIN_PATH, mode="train", load_cached_data=False
    )

    print(f"Extracted {len(train_features)} training feature rows.")

    # Assertions
    assert not train_features.empty, "Training features DataFrame is empty."
    assert (
        "target" in train_features.columns
    ), "Target column missing in training features."
    assert (
        "sim_k3_max_loc" in train_features.columns
    ), "Feature column sim_k3_max_loc missing."

    # Extract features for Val (Demo subset)
    val_features = engineer.extract_features(
        metadata_path=Config.VAL_PATH, mode="val", load_cached_data=False
    )
    assert not val_features.empty, "Validation features DataFrame is empty."

    print("Feature extraction verification passed.")
    return train_features, val_features


def test_ranker(train_df, val_df):
    """
    Demonstrates training the LightGBM ranker.
    """
    print("\n>>> Testing Ranker Training...")

    ranker = LGBMRanker()

    # Train the model
    ranker.train(train_df, val_df)

    # Verify model file
    assert os.path.exists(Config.LGBM_MODEL_PATH), "LGBM model file not created."

    # Test prediction
    preds = ranker.predict(val_df)
    assert len(preds) == len(val_df), "Prediction length mismatch."
    assert np.all(
        (preds >= -0.5) & (preds <= 1.5)
    ), "Predictions wildly out of expected normalized range [0,1]."

    print("Ranker training verification passed.")


def test_inference_pipeline():
    """
    Demonstrates the full inference pipeline (Val and Test).
    """
    print("\n>>> Testing Inference Pipeline...")

    pipeline = InferencePipeline()

    # 1. Validation Inference
    # This calculates Kendall Tau on the validation set
    # We use cached features generated in previous step to save time
    score = pipeline.run_validation_inference(load_cached_features=True)
    print(f"Demo Validation Score: {score}")
    assert isinstance(score, float), "Validation score should be a float."

    # 2. Test Inference
    # This generates the submission file
    pipeline.run_test_inference(load_cached_features=False)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert not df_sub.empty, "Submission file is empty."
    assert (
        "id" in df_sub.columns and "cell_order" in df_sub.columns
    ), "Submission columns missing."

    # Check if number of rows matches the demo test set size (20)
    # Note: If some notebooks have no markdown/code, they might be handled gracefully but id should exist.
    df_test_meta = pd.read_csv(Config.TEST_PATH)
    assert len(df_sub) == len(
        df_test_meta
    ), f"Expected {len(df_test_meta)} predictions, got {len(df_sub)}"

    print("Inference pipeline verification passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # 1. Setup
    setup_demo_environment()

    # 2. Test Utils
    test_utils()

    # 3. Test Backbone (Fine-Tuning)
    # Returns the trainer instance to reuse the loaded model
    trainer = test_backbone_fine_tuning()

    # 4. Test Feature Extraction
    # Returns dataframes for ranker training
    train_feats, val_feats = test_feature_extraction(trainer)

    # 5. Test Ranker
    test_ranker(train_feats, val_feats)

    # 6. Test Full Inference Pipeline
    test_inference_pipeline()

    print("\n>>> All demonstrations completed successfully.")
