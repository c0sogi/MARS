import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

# Import from provided library
import library.config as config
from library.dataset import StackExchangeDataset
from library.model import SiameseNetwork
from library.trainer import train_stage_1
from library.feature_caching import cache_features
from library.refinement import train_ridge_head
from library.inference import predict_and_submit


def run_demo():
    # -------------------------------------------------------------------------
    # 0. Setup
    # -------------------------------------------------------------------------
    print(">>> Setting up demonstration...")
    config.seed_everything(config.SEED)

    # Verify device
    print(f"Using device: {config.DEVICE}")

    # -------------------------------------------------------------------------
    # 1. Unit Test: Dataset and Model
    # -------------------------------------------------------------------------
    print("\n>>> [Demo] Instantiating Dataset and Model for verification...")

    # Create dummy data
    dummy_df = pd.DataFrame(
        {
            "qa_id": [1, 2],
            "question_title": ["How to code?", "What is ML?"],
            "question_body": ["I need help with coding.", "Explain machine learning."],
            "answer": ["Use a keyboard.", "It is statistics."],
        }
    )
    # Add dummy targets
    for col in config.TARGET_COLS:
        dummy_df[col] = np.random.rand(2)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Instantiate Dataset
    dataset = StackExchangeDataset(
        dummy_df,
        tokenizer,
        max_length=128,
        target_cols=config.TARGET_COLS,
        is_test=False,
    )

    # Verify Dataset Item
    item = dataset[0]
    required_keys = [
        "q_input_ids",
        "q_attention_mask",
        "a_input_ids",
        "a_attention_mask",
        "labels",
    ]
    for key in required_keys:
        assert key in item, f"Dataset item missing key: {key}"
    assert (
        item["labels"].shape[0] == 30
    ), f"Incorrect label shape: {item['labels'].shape}"
    print("Dataset verification passed.")

    # Instantiate Model
    model = SiameseNetwork(model_name=config.MODEL_NAME).to(config.DEVICE)

    # Verify Forward Pass
    q_ids = item["q_input_ids"].unsqueeze(0).to(config.DEVICE)
    q_mask = item["q_attention_mask"].unsqueeze(0).to(config.DEVICE)
    a_ids = item["a_input_ids"].unsqueeze(0).to(config.DEVICE)
    a_mask = item["a_attention_mask"].unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        logits = model(q_ids, q_mask, a_ids, a_mask)

    assert logits.shape == (
        1,
        30,
    ), f"Model output shape mismatch. Expected (1, 30), got {logits.shape}"
    print("Model forward pass verification passed.")

    # Clean up memory
    del model, dataset, tokenizer, dummy_df
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 2. Stage 1: Fine-Tuning (Debug Mode)
    # -------------------------------------------------------------------------
    print("\n>>> [Pipeline] Starting Stage 1: Fine-Tuning (Debug Mode)...")
    # We use debug=True to train on only 100 rows for speed
    best_score = train_stage_1(debug=True, load_cached_data=True)

    # Verify artifact creation
    assert os.path.exists(
        config.MODEL_STATE_DICT_PATH
    ), f"Model file not found at {config.MODEL_STATE_DICT_PATH}"
    print(f"Stage 1 complete. Model saved. Best debug score: {best_score:.4f}")

    # -------------------------------------------------------------------------
    # 3. Feature Caching (Full Data)
    # -------------------------------------------------------------------------
    print("\n>>> [Pipeline] Starting Feature Caching (Full Data)...")
    # We use debug=False here to ensure we extract features for the FULL dataset.
    # This is necessary because Stage 2 (Ridge) expects alignment with the full metadata files.
    # On A100, inference for ~6000 rows is very fast.
    cache_features(debug=False, load_cached_data=True)

    # Verify feature files
    expected_files = [
        config.TRAIN_FEATURES_PATH,
        config.TRAIN_TARGETS_PATH,
        config.VAL_FEATURES_PATH,
        config.TEST_FEATURES_PATH,
    ]
    for fpath in expected_files:
        assert os.path.exists(fpath), f"Feature file missing: {fpath}"
    print("Feature caching complete.")

    # -------------------------------------------------------------------------
    # 4. Stage 2: Ridge Regression
    # -------------------------------------------------------------------------
    print("\n>>> [Pipeline] Starting Stage 2: Ridge Regression...")
    train_ridge_head(load_cached_model=False)

    # Verify model artifact
    assert os.path.exists(
        config.RIDGE_MODEL_PATH
    ), f"Ridge model not found at {config.RIDGE_MODEL_PATH}"
    print("Stage 2 complete.")

    # -------------------------------------------------------------------------
    # 5. Inference
    # -------------------------------------------------------------------------
    print("\n>>> [Pipeline] Starting Inference...")
    predict_and_submit(load_cached_data=True, debug=False)

    # Verify Submission
    assert os.path.exists(
        config.SUBMISSION_PATH
    ), f"Submission file not found at {config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Check shape (Test set has 608 rows)
    assert len(df_sub) == 608, f"Submission has incorrect row count: {len(df_sub)}"
    assert (
        df_sub.shape[1] == 31
    ), f"Submission has incorrect column count: {df_sub.shape[1]}"

    # Check values
    target_vals = df_sub[config.TARGET_COLS].values
    assert np.all(target_vals >= 0) and np.all(
        target_vals <= 1
    ), "Predictions contain values outside [0, 1] range."

    print("Inference verification passed.")
    print("\n>>> Demonstration completed successfully!")


if __name__ == "__main__":
    run_demo()
