import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil
from transformers import AutoTokenizer

# Add current directory to path to allow imports from library
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config, seed_everything
from library.utils import jaccard, AverageMeter
from library.dataset import process_data, TweetDataset
from library.model import TweetModel
from library.engine import run_training, run_inference


def run_demo():
    # ==========================================
    # 1. Setup and Reproducibility
    # ==========================================
    print("1. Setting up environment and seeds...")
    seed_everything(Config.SEED)

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # 2. Verify Utility Functions
    # ==========================================
    print("\n2. Verifying Utility Functions...")

    # Test Jaccard Similarity
    s1, s2, s3 = "hello world", "hello world", "hello"
    score_perfect = jaccard(s1, s2)
    score_partial = jaccard(s1, s3)

    print(f"   Jaccard('{s1}', '{s2}') = {score_perfect}")
    print(f"   Jaccard('{s1}', '{s3}') = {score_partial:.4f}")

    assert score_perfect == 1.0, "Jaccard calculation failed for identical strings"
    assert 0.0 < score_partial < 1.0, "Jaccard calculation failed for partial overlap"

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=1)
    meter.update(20, n=1)
    assert meter.avg == 15.0, "AverageMeter failed calculation"
    print("   Utility functions verified.")

    # ==========================================
    # 3. Verify Data Processing and Dataset Class
    # ==========================================
    print("\n3. Verifying Data Processing and Dataset Class...")

    # Load a small slice of training data from metadata
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    # Filter out neutral for training logic check (mimicking engine.py strategy)
    df_train_filt = df_train[df_train["sentiment"] != "neutral"].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    # Run process_data in debug mode (processes first 100 rows)
    # This generates cache files in ./working/idea_1
    print("   Processing data (debug mode)...")
    data_dict = process_data(
        df_train_filt,
        tokenizer,
        max_len=Config.MAX_LEN,
        cache_dir=Config.WORKING_DIR,
        prefix="demo_train",
        load_cached_data=False,  # Force processing for demo
        debug=True,
    )

    # Verify keys in data_dict
    expected_keys = [
        "ids",
        "mask",
        "offsets",
        "orig_tweet",
        "sentiment",
        "text_ids",
        "targets_start",
        "targets_end",
    ]
    for k in expected_keys:
        assert k in data_dict, f"Missing key {k} in processed data"

    print(f"   Processed data shapes: IDs {data_dict['ids'].shape}")

    # Instantiate Dataset
    dataset = TweetDataset(data_dict)
    assert len(dataset) > 0, "Dataset is empty"

    # Get one item to verify structure
    item = dataset[0]

    # Verify Tensor shapes and types
    assert item["ids"].shape[0] == Config.MAX_LEN, "Incorrect input_ids length"
    assert item["mask"].shape[0] == Config.MAX_LEN, "Incorrect attention_mask length"
    assert isinstance(item["ids"], torch.Tensor), "ids is not a Tensor"
    assert (
        "targets_start" in item and "targets_end" in item
    ), "Targets missing from training item"

    print("   Dataset class verified.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n4. Verifying Model Architecture...")
    device = Config.DEVICE
    model = TweetModel(Config)
    model.to(device)
    model.eval()

    # Create a dummy batch [Batch Size=1, Seq Len]
    input_ids = item["ids"].unsqueeze(0).to(device)
    mask = item["mask"].unsqueeze(0).to(device)

    print("   Running forward pass...")
    with torch.no_grad():
        start_logits, end_logits = model(input_ids, mask)

    print(f"   Model Output Shapes: Start {start_logits.shape}, End {end_logits.shape}")

    # Verify output shapes
    assert start_logits.shape == (1, Config.MAX_LEN), "Incorrect start_logits shape"
    assert end_logits.shape == (1, Config.MAX_LEN), "Incorrect end_logits shape"
    print("   Model architecture verified.")

    # ==========================================
    # 5. Verify Training Engine
    # ==========================================
    print("\n5. Verifying Training Engine (Debug Mode)...")

    # Ensure clean slate for model file
    if os.path.exists(Config.MODEL_SAVE_PATH):
        os.remove(Config.MODEL_SAVE_PATH)

    # Run training for 1 epoch on debug subset
    try:
        run_training(epochs=1, patience=1, debug=True)
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Check if model checkpoint was created
    # Note: In a tiny debug run, validation might not trigger a save if metric is 0,
    # but we ensure a model exists for the inference step.
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(
            "   Note: Model not saved by training loop (likely due to metric). Saving manually for inference test."
        )
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    else:
        print(f"   Training successful. Model saved at {Config.MODEL_SAVE_PATH}")

    # ==========================================
    # 6. Verify Inference Engine
    # ==========================================
    print("\n6. Verifying Inference Engine (Debug Mode)...")

    # Ensure clean slate for submission
    if os.path.exists(Config.SUBMISSION_PATH):
        os.remove(Config.SUBMISSION_PATH)

    try:
        run_inference(debug=True)
    except Exception as e:
        print(f"Inference failed with error: {e}")
        raise e

    # Check submission file integrity
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"   Inference successful. Submission shape: {sub_df.shape}")

        assert "textID" in sub_df.columns
        assert "selected_text" in sub_df.columns
        assert len(sub_df) > 0, "Submission file is empty"

        # Check for quoted text format implicitly by checking content
        print(f"   Sample prediction: {sub_df.iloc[0]['selected_text']}")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\nAll demonstrations and verifications passed successfully.")


if __name__ == "__main__":
    run_demo()
