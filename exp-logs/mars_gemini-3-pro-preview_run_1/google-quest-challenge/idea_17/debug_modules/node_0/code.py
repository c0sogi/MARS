import os
import shutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_spearman_correlation
from library.dataset import StackExchangeDataset, Collate
from library.model import DistilRoBERTaDualEncoder
from library.engine import Engine


def run_demo():
    print("--- Starting Demo & Verification ---")

    # 1. Setup Configuration for Speed and Isolation
    # We override Config attributes to use a temp directory and run quickly
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Setting up configuration in {DEMO_DIR}...")

    # Override Config
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Speed optimizations
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Set seeds
    seed_everything(42)

    # 2. Prepare Subset Data
    # We read the original metadata, sample 20 rows, and save to DEMO_DIR.
    # We then point Config to these new files.
    print("Creating data subsets...")

    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Take top 20 rows
    subset_train = orig_train.head(20).copy()
    subset_val = orig_val.head(20).copy()
    subset_test = orig_test.head(20).copy()

    # Save to demo dir
    train_path = os.path.join(DEMO_DIR, "train_subset.csv")
    val_path = os.path.join(DEMO_DIR, "val_subset.csv")
    test_path = os.path.join(DEMO_DIR, "test_subset.csv")

    subset_train.to_csv(train_path, index=False)
    subset_val.to_csv(val_path, index=False)
    subset_test.to_csv(test_path, index=False)

    # Update Config paths
    Config.TRAIN_PATH = train_path
    Config.VAL_PATH = val_path
    Config.TEST_PATH = test_path

    # 3. Verify Metric Logic
    print("Verifying Metric (Spearman Correlation)...")
    # Case A: Perfect correlation
    t1 = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7]])
    p1 = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7]])
    score_perfect = compute_spearman_correlation(p1, t1)
    assert np.isclose(score_perfect, 1.0), f"Expected 1.0, got {score_perfect}"

    # Case B: Inverse correlation
    p2 = np.array([[0.3, 0.7], [0.2, 0.6], [0.1, 0.5]])
    score_inverse = compute_spearman_correlation(p2, t1)
    assert np.isclose(score_inverse, -1.0), f"Expected -1.0, got {score_inverse}"
    print("Metric verification passed.")

    # 4. Verify Dataset and Collate
    print("Verifying Dataset and Collate...")
    engine_instance = Engine()  # Initializes tokenizer
    tokenizer = engine_instance.tokenizer

    ds = StackExchangeDataset(subset_train, tokenizer, max_len=128, is_test=False)
    collate_fn = Collate(tokenizer)
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)

    batch = next(iter(dl))

    # Check keys
    expected_keys = {
        "qa_id",
        "q_input_ids",
        "q_attention_mask",
        "a_input_ids",
        "a_attention_mask",
        "labels",
    }
    assert (
        set(batch.keys()) == expected_keys
    ), f"Missing keys in batch. Found: {batch.keys()}"

    # Check shapes
    assert batch["q_input_ids"].dim() == 2
    assert batch["q_input_ids"].shape[0] == 4
    assert batch["labels"].shape == (4, 30)
    print("Dataset and Collate verification passed.")

    # 5. Verify Model Logic
    print("Verifying Model...")
    model = DistilRoBERTaDualEncoder()
    model.to(Config.DEVICE)

    # Forward pass check
    # Move batch to device
    device_batch = {k: v.to(Config.DEVICE) for k, v in batch.items()}

    with torch.no_grad():
        logits = model(
            q_input_ids=device_batch["q_input_ids"],
            q_attention_mask=device_batch["q_attention_mask"],
            a_input_ids=device_batch["a_input_ids"],
            a_attention_mask=device_batch["a_attention_mask"],
        )

    assert logits.shape == (4, 30), f"Expected output shape (4, 30), got {logits.shape}"

    # Freeze/Unfreeze check
    model.freeze_backbone()
    for name, param in model.q_backbone.named_parameters():
        assert param.requires_grad is False, f"Backbone param {name} should be frozen"

    model.unfreeze_backbone()
    # Check one param to ensure it's unfrozen
    for name, param in model.q_backbone.named_parameters():
        if "embeddings" in name:  # Check an embedding layer
            assert (
                param.requires_grad is True
            ), f"Backbone param {name} should be unfrozen"
            break

    print("Model verification passed.")

    # 6. Run Engine (Integration Test)
    print("Running Engine integration test (Train/Val/Predict)...")
    # This will use the subset data configured in Config
    engine_instance.run()

    # 7. Verify Output
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with shape: {sub_df.shape}")

    # Expect 20 rows (from subset_test) + header
    assert len(sub_df) == 20, f"Expected 20 predictions, got {len(sub_df)}"
    assert (
        sub_df.shape[1] == 31
    ), f"Expected 31 columns (qa_id + 30 targets), got {sub_df.shape[1]}"
    assert "qa_id" in sub_df.columns

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
