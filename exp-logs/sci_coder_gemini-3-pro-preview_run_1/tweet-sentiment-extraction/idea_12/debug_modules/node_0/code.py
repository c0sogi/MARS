import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import from provided library
from library.config import Config
from library.utils import seed_everything, normalize_text, jaccard, AverageMeter
from library.data import get_data_loaders, get_test_loader, TweetDataset
from library.model import TweetModel
from library.engine import loss_fn, get_optimizer_params, run_experiment

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    # =========================================================================
    # 1. Configuration Overrides for Speed & Demo
    # =========================================================================
    print(">>> [1/6] Configuring Environment...")

    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.N_FOLDS = 1  # Run only 1 fold (Engine breaks after 1 anyway)
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8

    # Use a separate working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution/"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.bin")

    # Ensure directories exist and are clean
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # =========================================================================
    # 2. Testing Utilities
    # =========================================================================
    print("\n>>> [2/6] Verifying Utilities...")
    seed_everything(Config.SEED)

    # Test Text Normalization
    raw_text = "   Hello   World!  "
    clean_text = normalize_text(raw_text)
    assert clean_text == "Hello World!", f"Normalization failed: '{clean_text}'"

    # Test Jaccard Score
    s1 = "sentiment analysis"
    s2 = "sentiment analysis is fun"
    # Intersection: {sentiment, analysis} (2)
    # Union: {sentiment, analysis, is, fun} (4)
    # Jaccard: 2/4 = 0.5
    score = jaccard(s1, s2)
    assert abs(score - 0.5) < 1e-6, f"Jaccard calculation failed: {score}"

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(val=1.0, n=1)
    meter.update(val=2.0, n=1)
    assert meter.avg == 1.5, f"AverageMeter failed: {meter.avg}"

    print("    Utils verified successfully.")

    # =========================================================================
    # 3. Testing Data Pipeline
    # =========================================================================
    print("\n>>> [3/6] Verifying Data Pipeline...")

    # Generate DataLoaders (this will trigger processing and caching)
    # We set load_cached_data=False to ensure we test the processing logic
    train_loader, val_loader = get_data_loaders(
        fold=0, load_cached_data=False, debug=True
    )

    # Verify Batch Structure
    batch = next(iter(train_loader))
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_targets",
        "end_targets",
        "offsets",
    ]
    for key in required_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Verify Shapes
    # Input IDs: (Batch, Seq_Len)
    assert batch["input_ids"].shape == (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN)
    # Targets: (Batch, Seq_Len)
    assert batch["start_targets"].shape == (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN)

    print(f"    Batch Size: {batch['input_ids'].shape[0]}")
    print(f"    Sequence Length: {batch['input_ids'].shape[1]}")
    print("    Data Pipeline verified successfully.")

    # =========================================================================
    # 4. Testing Model Architecture
    # =========================================================================
    print("\n>>> [4/6] Verifying Model Architecture...")

    device = Config.DEVICE
    model = TweetModel()
    model.to(device)

    # Perform Forward Pass
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    start_logits, end_logits = model(input_ids, attention_mask)

    # Verify Output Shapes
    assert start_logits.shape == (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN)
    assert end_logits.shape == (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN)

    print("    Forward pass successful.")
    print(f"    Logits Shape: {start_logits.shape}")

    # =========================================================================
    # 5. Testing Loss & Optimization
    # =========================================================================
    print("\n>>> [5/6] Verifying Loss & Optimizer...")

    start_targets = batch["start_targets"].to(device)
    end_targets = batch["end_targets"].to(device)

    # Compute Loss
    loss = loss_fn(start_logits, end_logits, start_targets, end_targets)
    assert not torch.isnan(loss).any(), "Loss computation resulted in NaN"
    print(f"    Initial Loss: {loss.item():.4f}")

    # Verify Optimizer Parameters
    params = get_optimizer_params(model)
    assert len(params) > 0, "No parameters returned for optimizer"
    print(f"    Optimizer parameter groups: {len(params)}")

    # =========================================================================
    # 6. Running Full Experiment
    # =========================================================================
    print("\n>>> [6/6] Executing Full Experiment Loop...")
    print("    This runs training, validation, and inference using the Engine.")

    # Free memory before full run
    del model, batch, input_ids, attention_mask, start_logits, end_logits
    torch.cuda.empty_cache()

    # Execute the engine's main function
    # This uses the Config overrides we set at the beginning
    run_experiment()

    # Verify Output
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"\n    Submission generated at: {Config.SUBMISSION_PATH}")
        print(f"    Rows: {len(df_sub)}")
        print(f"    Columns: {list(df_sub.columns)}")

        # Basic content check
        assert "textID" in df_sub.columns
        assert "selected_text" in df_sub.columns
        assert len(df_sub) > 0
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n>>> Demo Execution Completed Successfully.")


if __name__ == "__main__":
    run_demo()
