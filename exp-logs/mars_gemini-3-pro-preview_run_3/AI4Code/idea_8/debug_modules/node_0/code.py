import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
import logging

# Suppress warnings and logs for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)

# Import library components
from library.config import Config
from library.utils import set_seed, calc_kendall_tau, reconstruct_order
from library.semantic_model import FineTuner
from library.feature_engineering import generate_features
from library.ranker_model import LGBMRanker


def run_demo():
    print("Starting Library Demo...")

    # =========================================================================
    # 1. Configuration Override for Speed and Isolation
    # =========================================================================
    print("Configuring environment...")

    # Enable Debug mode to use a tiny subset of data (e.g., 5 notebooks)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5

    # Reduce training parameters for the demo
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.WARMUP_STEPS = 0

    # Set a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update output paths to point to the demo directory
    Config.BACKBONE_OUTPUT_DIR = os.path.join(Config.WORKING_DIR, "dsapr_model")
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )
    Config.LGBM_MODEL_PATH = os.path.join(Config.WORKING_DIR, "lgbm_model.txt")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure reproducibility
    set_seed(Config.SEED)

    # =========================================================================
    # 2. Semantic Model Fine-Tuning
    # =========================================================================
    print("\n--- Step 2: Semantic Model Fine-Tuning ---")

    # Initialize the FineTuner
    fine_tuner = FineTuner()

    # Run training (this will generate pairs from the debug subset)
    # forcing load_cached_data=False to demonstrate data generation
    fine_tuner.train(load_cached_data=False)

    # Verify model was saved
    if not os.path.exists(Config.BACKBONE_OUTPUT_DIR):
        raise FileNotFoundError("Fine-tuned model artifact not found after training.")
    print("Semantic model fine-tuning and saving verified.")

    # =========================================================================
    # 3. Feature Engineering
    # =========================================================================
    print("\n--- Step 3: Feature Generation ---")

    # Generate features for Training set
    print("Generating training features...")
    df_train_feats = generate_features(
        metadata_path=Config.TRAIN_METADATA_PATH,
        output_path=Config.TRAIN_FEATURES_PATH,
        load_cached_data=False,  # Force regeneration
        debug=True,
    )

    # Generate features for Validation set
    print("Generating validation features...")
    df_val_feats = generate_features(
        metadata_path=Config.VAL_METADATA_PATH,
        output_path=Config.VAL_FEATURES_PATH,
        load_cached_data=False,
        debug=True,
    )

    # Verification
    assert not df_train_feats.empty, "Training features DataFrame is empty."
    assert not df_val_feats.empty, "Validation features DataFrame is empty."
    required_cols = ["n_code", "md_len", "smoothed_best_match_loc", "sim_max", "rank"]
    for col in required_cols:
        assert col in df_train_feats.columns, f"Missing column {col} in features."

    print(
        f"Generated {len(df_train_feats)} training rows and {len(df_val_feats)} validation rows."
    )

    # =========================================================================
    # 4. Ranker Model Training
    # =========================================================================
    print("\n--- Step 4: Ranker Training (LightGBM) ---")

    ranker = LGBMRanker()
    ranker.train(df_train_feats, df_val_feats)

    if not os.path.exists(Config.LGBM_MODEL_PATH):
        raise FileNotFoundError("LightGBM model file not found after training.")
    print("Ranker training verified.")

    # =========================================================================
    # 5. Inference and Metric Evaluation
    # =========================================================================
    print("\n--- Step 5: Inference and Evaluation ---")

    # Predict on validation set
    val_preds = ranker.predict(df_val_feats)

    # Assign predictions back to the dataframe
    df_val_feats["pred_rank"] = val_preds

    # Reconstruct order for a single notebook to demonstrate the process
    sample_nb_id = df_val_feats["id"].iloc[0]
    print(f"Evaluating sample notebook: {sample_nb_id}")

    # Get ground truth order from metadata
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    gt_row = df_val_meta[df_val_meta["id"] == sample_nb_id].iloc[0]
    gt_order_str = gt_row["cell_order"]
    gt_order_list = gt_order_str.split()

    # Get code cells from the notebook (needed for reconstruction)
    # We can infer code cells by filtering out markdown cells from the GT list
    # or by reading the file. Let's read the file using library utils.
    from library.utils import read_notebook

    cell_types, _ = read_notebook(os.path.join(Config.INPUT_DIR, gt_row["file_path"]))

    code_cells = [cid for cid in gt_order_list if cell_types.get(cid) == "code"]

    # Get predicted scores for markdown cells in this notebook
    nb_preds = df_val_feats[df_val_feats["id"] == sample_nb_id]
    markdown_scores = dict(zip(nb_preds["cell_id"], nb_preds["pred_rank"]))

    # Reconstruct order
    predicted_order_str = reconstruct_order(code_cells, markdown_scores)

    # Calculate Metric
    # Create submission dict format
    submission_dict = {sample_nb_id: predicted_order_str}

    # Create a mini GT dataframe for this sample
    df_sample_gt = pd.DataFrame([{"id": sample_nb_id, "cell_order": gt_order_str}])

    score = calc_kendall_tau(df_sample_gt, submission_dict)

    print(f"Predicted Order (First 50 chars): {predicted_order_str[:50]}...")
    print(f"Kendall Tau Score for sample: {score:.4f}")

    # Final assertion
    assert -1.0 <= score <= 1.0, f"Score {score} is out of valid range [-1, 1]"

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
