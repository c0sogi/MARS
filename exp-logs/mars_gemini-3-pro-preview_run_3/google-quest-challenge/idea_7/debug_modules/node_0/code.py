import os
import shutil
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig

# ------------------------------------------------------------------------------
# 1. Configuration & Setup
# ------------------------------------------------------------------------------
# Import Config first to modify it for the demo
from library.config import Config

print("Configuring demo run...")

# Override Config for speed and resource efficiency
Config.WORKING_DIR = "./working/demo_run"
Config.MODEL_DEBERTA = "prajjwal1/bert-tiny"  # Use a tiny model for rapid execution
Config.EPOCHS = 1
Config.DAPT_EPOCHS = 1
Config.N_FOLDS = 2
Config.TRAIN_BATCH_SIZE = 8
Config.VALID_BATCH_SIZE = 16
Config.MAX_LEN = 32  # Short sequence length for speed
Config.DAPT_BATCH_SIZE = 8
Config.DAPT_MODEL_OUTPUT_PATH = os.path.join(Config.WORKING_DIR, "dapt_demo_model")
Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

# Clean working directory
if os.path.exists(Config.WORKING_DIR):
    shutil.rmtree(Config.WORKING_DIR)
os.makedirs(Config.WORKING_DIR)

# Import library modules after Config modification
from library import dataset, dapt, fine_tuning, stacking, utils


# Monkey-patch get_tokenizer to ensure it uses the tiny model
# (The original function's default arg is frozen at import time)
def demo_get_tokenizer(model_name=Config.MODEL_DEBERTA):
    return AutoTokenizer.from_pretrained(model_name)


dataset.get_tokenizer = demo_get_tokenizer

# ------------------------------------------------------------------------------
# 2. Main Execution Pipeline
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    utils.set_seed(Config.SEED)

    print("\n" + "=" * 40)
    print(" Step 1: Domain-Adaptive Pre-Training (DAPT)")
    print("=" * 40)

    # Run DAPT (Masked Language Modeling)
    # This will save the adapted model to Config.DAPT_MODEL_OUTPUT_PATH
    dapt.run_dapt(load_cached_data=False)

    # Verify DAPT output
    assert os.path.exists(
        Config.DAPT_MODEL_OUTPUT_PATH
    ), "DAPT output directory missing."
    assert os.path.exists(
        os.path.join(Config.DAPT_MODEL_OUTPUT_PATH, "config.json")
    ), "DAPT model config missing."
    print("DAPT completed and verified.")

    print("\n" + "=" * 40)
    print(" Step 2: Supervised Fine-Tuning & Feature Extraction")
    print("=" * 40)

    # Run Fine-Tuning
    # This runs 2 folds (Config.N_FOLDS) and extracts features for train (OOF), val, and test.
    model_alias = "demo_model"
    fine_tuning.run_fine_tuning(
        model_alias=model_alias,
        base_model_name=Config.MODEL_DEBERTA,
        dapt_path=Config.DAPT_MODEL_OUTPUT_PATH,
        load_cached_data=False,
    )

    # Verify Fine-Tuning output features
    train_feats_path = os.path.join(
        Config.WORKING_DIR, f"{model_alias}_train_features.npy"
    )
    test_feats_path = os.path.join(
        Config.WORKING_DIR, f"{model_alias}_test_features.npy"
    )

    assert os.path.exists(train_feats_path), "Train features file missing."
    assert os.path.exists(test_feats_path), "Test features file missing."

    # Check feature dimensions
    # CustomBackbone outputs [h_cls, h_q, h_a, h_diff], so dim is 4 * hidden_size
    # prajjwal1/bert-tiny has hidden_size=128
    tiny_config = AutoConfig.from_pretrained(Config.MODEL_DEBERTA)
    expected_dim = 4 * tiny_config.hidden_size

    train_feats = np.load(train_feats_path)
    print(f"Train features shape: {train_feats.shape}")

    assert train_feats.ndim == 2, "Features should be 2D array."
    assert (
        train_feats.shape[1] == expected_dim
    ), f"Expected feature dim {expected_dim}, got {train_feats.shape[1]}"
    print("Fine-tuning features verified.")

    print("\n" + "=" * 40)
    print(" Step 3: Stacking (Level 1 & Level 2)")
    print("=" * 40)

    # Train Meta Stacker
    # This trains L1 ridge models on the extracted features, then an L2 ridge on the predictions.
    stacking.train_meta_stacker(model_aliases=[model_alias], load_cached_preds=False)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file missing."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")

    # Check shape: (608 test rows, 31 columns including qa_id)
    # Note: We rely on the provided test_metadata.csv length
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    expected_rows = len(test_meta)
    expected_cols = 31  # qa_id + 30 targets

    assert sub_df.shape == (
        expected_rows,
        expected_cols,
    ), f"Submission shape mismatch. Expected ({expected_rows}, {expected_cols}), got {sub_df.shape}"

    # Check value range [0, 1]
    # Exclude qa_id column
    preds = sub_df.iloc[:, 1:].values
    assert preds.min() >= 0.0, "Predictions contain values < 0."
    assert preds.max() <= 1.0, "Predictions contain values > 1."

    print("Submission file verified successfully.")
    print("\nDemo execution completed.")
