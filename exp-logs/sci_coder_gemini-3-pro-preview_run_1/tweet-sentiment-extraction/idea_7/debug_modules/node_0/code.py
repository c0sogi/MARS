import os
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import components from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_data, TweetDataset
from library.model import TweetModel
from library.engine import train_fn, eval_fn


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Setup Configuration for Demo
    # We subclass Config to override parameters for a quick run
    class DemoConfig(Config):
        TRAIN_BATCH_SIZE = 4
        VALID_BATCH_SIZE = 4
        EPOCHS = 1
        MAX_LEN = 64  # Reduced length for speed
        CACHE_DIR = "./working/demo_cache"  # Isolated cache for demo
        # Ensure we use the provided model path
        MODEL_PATH = "microsoft/deberta-v3-base"

    config = DemoConfig()

    # Set random seed for reproducibility
    seed_everything(config.SEED)

    # Clean up any existing demo cache
    if os.path.exists(config.CACHE_DIR):
        shutil.rmtree(config.CACHE_DIR)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # 2. Load and Subset Data
    print("\n[1] Loading Metadata...")
    if not os.path.exists(config.TRAIN_META):
        raise FileNotFoundError(f"Metadata file not found: {config.TRAIN_META}")

    df_raw = pd.read_csv(config.TRAIN_META)
    # Select a small subset of non-neutral tweets (neutral usually just returns full text)
    # We use non-neutral to verify the start/end index logic works
    df_subset = df_raw[df_raw["sentiment"] != "neutral"].head(16).reset_index(drop=True)
    print(f"    Created subset of {len(df_subset)} samples.")

    # 3. Data Processing Verification
    print("\n[2] Verifying Data Processing (get_data)...")
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)

    # Process data (this handles tokenization, offsets, and target generation)
    # cache_name='demo_train' implies training mode (targets are generated)
    data = get_data(
        df_subset, tokenizer, config, cache_name="demo_train", load_cached_data=False
    )

    expected_keys = [
        "input_ids",
        "attention_mask",
        "offsets",
        "start_indices",
        "end_indices",
    ]
    for key in expected_keys:
        assert key in data, f"Missing key in processed data: {key}"
        assert len(data[key]) == len(df_subset), f"Length mismatch for {key}"

    print("    Data processing successful. Keys and shapes verified.")

    # 4. Dataset and DataLoader Verification
    print("\n[3] Verifying Dataset and DataLoader...")
    dataset = TweetDataset(data, config, is_test=False)
    loader = DataLoader(dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=False)

    # Fetch a single batch
    batch = next(iter(loader))

    # Check shapes
    assert batch["input_ids"].shape == (config.TRAIN_BATCH_SIZE, config.MAX_LEN)
    assert batch["attention_mask"].shape == (config.TRAIN_BATCH_SIZE, config.MAX_LEN)
    assert batch["start_targets"].shape == (config.TRAIN_BATCH_SIZE, config.MAX_LEN)
    assert batch["end_targets"].shape == (config.TRAIN_BATCH_SIZE, config.MAX_LEN)

    # Verify Gaussian Smoothing properties (Probability distribution should sum to ~1)
    start_target_sums = batch["start_targets"].sum(dim=1)
    assert torch.allclose(
        start_target_sums, torch.ones_like(start_target_sums), atol=1e-4
    ), "Start targets do not sum to 1.0"

    print("    Dataset yields correct shapes and valid probability distributions.")

    # 5. Model Initialization and Forward Pass
    print("\n[4] Verifying Model and Forward Pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")

    model = TweetModel(config)
    model.to(device)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Run forward pass
    start_logits, end_logits = model(input_ids, attention_mask)

    assert start_logits.shape == (config.TRAIN_BATCH_SIZE, config.MAX_LEN)
    assert end_logits.shape == (config.TRAIN_BATCH_SIZE, config.MAX_LEN)

    print("    Forward pass successful. Logit shapes correct.")

    # 6. Training Loop Verification (Engine)
    print("\n[5] Verifying Training Engine (train_fn)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(loader) * config.EPOCHS
    )

    # Run one epoch of training on the subset
    loss = train_fn(loader, model, optimizer, device, scheduler)

    print(f"    Training step complete. Average Loss: {loss:.4f}")
    assert not np.isnan(loss), "Training produced NaN loss."

    # 7. Evaluation Loop Verification (Engine)
    print("\n[6] Verifying Evaluation Engine (eval_fn)...")
    # We use the same subset as validation for demonstration
    jaccard_score = eval_fn(loader, model, device, df_subset)

    print(f"    Evaluation complete. Jaccard Score: {jaccard_score:.4f}")
    assert 0.0 <= jaccard_score <= 1.0, "Jaccard score is out of valid range [0, 1]."

    # 8. Inference / Test Mode Verification
    print("\n[7] Verifying Inference Mode...")
    # Simulate test data by dropping the target column
    df_test = df_subset.drop(columns=["selected_text"])

    # 'test' in cache_name triggers is_test=True logic in get_data
    test_data = get_data(
        df_test, tokenizer, config, cache_name="demo_test", load_cached_data=False
    )

    # Verify targets are NOT present
    assert "start_indices" not in test_data

    test_dataset = TweetDataset(test_data, config, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=config.VALID_BATCH_SIZE)

    model.eval()
    with torch.no_grad():
        test_batch = next(iter(test_loader))
        t_input_ids = test_batch["input_ids"].to(device)
        t_mask = test_batch["attention_mask"].to(device)
        t_s_logits, t_e_logits = model(t_input_ids, t_mask)

    assert t_s_logits.shape == (config.VALID_BATCH_SIZE, config.MAX_LEN)
    print("    Inference mode successful.")

    # Cleanup
    if os.path.exists(config.CACHE_DIR):
        shutil.rmtree(config.CACHE_DIR)

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
