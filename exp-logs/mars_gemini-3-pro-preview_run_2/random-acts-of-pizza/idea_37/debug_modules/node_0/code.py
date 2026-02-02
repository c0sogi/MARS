import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Add current directory to path to ensure library imports work
sys.path.append(".")

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_dataset
from library.feature_engine import EmbeddingGenerator
from library.processors import FoldProcessor
from library.model_definitions import get_bagged_lr_pipeline, get_hyperparameter_grid
from library.trainer import CrossValidator

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Setup and Configuration Patching
    # We modify Config attributes to ensure the demo runs quickly and uses a separate workspace
    print("[1] Configuring environment for demo...")
    set_seed(42)

    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Patch Config
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce computational load for demo
    Config.N_SPLITS = 2
    Config.N_ESTIMATORS_BAGGING = 2
    Config.PCA_COMPONENTS = 5  # Reduced from 50 because sample size will be small
    Config.LR_PARAM_GRID = {"C": [0.1, 1.0]}  # Minimal grid

    logger = setup_logger(os.path.join(DEMO_DIR, "demo.log"), name="demo_logger")

    # 2. Data Loading
    print("[2] Loading and subsetting data...")
    # Load full datasets (ignoring cache to demonstrate loading logic)
    df_train_full, df_val_full, df_test_full = load_dataset(load_cached_data=False)

    # Create small subsets for speed
    N_TRAIN = 25
    N_VAL = 10
    N_TEST = 10

    df_train_sub = df_train_full.head(N_TRAIN).copy().reset_index(drop=True)
    df_val_sub = df_val_full.head(N_VAL).copy().reset_index(drop=True)
    df_test_sub = df_test_full.head(N_TEST).copy().reset_index(drop=True)

    print(
        f"    Subset shapes: Train={df_train_sub.shape}, Val={df_val_sub.shape}, Test={df_test_sub.shape}"
    )

    # Verify required columns exist
    assert (
        "text_combined" in df_train_sub.columns
    ), "Text preprocessing failed: 'text_combined' missing."
    assert (
        Config.TARGET_COL in df_train_sub.columns
    ), "Target column missing in training data."

    # 3. Feature Engineering (Embeddings)
    print("[3] Generating embeddings for subsets...")
    embedder = EmbeddingGenerator()

    # We manually call generate_dataset_embeddings with our subsets
    # Note: This will download models if not present, which might take a moment
    embeddings = embedder.generate_dataset_embeddings(
        df_train_sub, df_val_sub, df_test_sub, load_cached_data=False, batch_size=8
    )

    # Verify embedding structure
    assert "anchor" in embeddings and "aux" in embeddings
    assert "train" in embeddings["anchor"]

    anchor_train = embeddings["anchor"]["train"]
    aux_train = embeddings["aux"]["train"]
    anchor_val = embeddings["anchor"]["val"]
    aux_val = embeddings["aux"]["val"]
    anchor_test = embeddings["anchor"]["test"]
    aux_test = embeddings["aux"]["test"]

    # Verify shapes
    print(f"    Anchor Train Shape: {anchor_train.shape}")
    assert anchor_train.shape[0] == N_TRAIN
    assert anchor_train.shape[1] == 384  # MiniLM dimension
    assert aux_train.shape[1] == 768  # MPNet dimension

    # 4. Processor Usage (Fit/Transform)
    print("[4] Running FoldProcessor...")
    processor = FoldProcessor()

    # Fit on training subset
    views_train = processor.fit_transform(df_train_sub, anchor_train, aux_train)

    # Transform validation and test subsets
    views_val = processor.transform(df_val_sub, anchor_val, aux_val)
    views_test = processor.transform(df_test_sub, anchor_test, aux_test)

    # Verify Views
    print("    Verifying view generation...")
    # View A: Anchor (384) + Metadata (9) = 393 roughly
    # View B: Anchor (384) + PCA Aux (5) + Metadata (9) = 398 roughly

    feat_dim_A = views_train["view_A"].shape[1]
    feat_dim_B = views_train["view_B"].shape[1]

    print(f"    View A dims: {feat_dim_A}, View B dims: {feat_dim_B}")

    assert "view_A" in views_train
    assert "view_B" in views_train
    assert views_train["view_A"].shape[0] == N_TRAIN
    assert views_val["view_A"].shape[0] == N_VAL
    assert views_test["view_B"].shape[0] == N_TEST

    # Check for NaNs
    assert not np.isnan(views_train["view_A"]).any(), "NaNs found in training View A"
    assert not np.isnan(views_val["view_B"]).any(), "NaNs found in validation View B"

    # 5. Model Training (Simulating Trainer Logic)
    print("[5] Training Model (Bagged LR with GridSearch)...")

    # We use the CrossValidator's helper method logic manually here
    # to avoid running the full CV loop which re-loads data
    cv_trainer = CrossValidator()

    y_train = df_train_sub[Config.TARGET_COL].values.astype(int)

    # Train on View A
    print("    Training on View A...")
    model_a = cv_trainer.tune_and_train(
        views_train["view_A"], y_train, pipeline_name="Demo_Pipeline_A"
    )

    # Train on View B
    print("    Training on View B...")
    model_b = cv_trainer.tune_and_train(
        views_train["view_B"], y_train, pipeline_name="Demo_Pipeline_B"
    )

    # Verify models are fitted
    from sklearn.utils.validation import check_is_fitted

    check_is_fitted(model_a)
    check_is_fitted(model_b)

    # 6. Inference and Consensus
    print("[6] Performing Inference...")

    # Predict on Test
    prob_a = model_a.predict_proba(views_test["view_A"])[:, 1]
    prob_b = model_b.predict_proba(views_test["view_B"])[:, 1]

    # Consensus
    consensus_preds = 0.5 * prob_a + 0.5 * prob_b

    print(f"    Predictions (first 5): {consensus_preds[:5]}")

    assert len(consensus_preds) == N_TEST
    assert (consensus_preds >= 0).all() and (consensus_preds <= 1).all()

    # 7. Save Submission
    print("[7] Saving Submission...")
    cv_trainer.save_submission(df_test_sub, consensus_preds)

    assert os.path.exists(Config.SUBMISSION_PATH)

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert df_sub.shape == (N_TEST, 2)
    assert list(df_sub.columns) == ["request_id", "requester_received_pizza"]

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
