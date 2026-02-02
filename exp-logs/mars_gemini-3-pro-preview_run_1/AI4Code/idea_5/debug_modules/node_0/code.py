import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, compute_kendall_tau
from library.preprocess import precompute_features
from library.dataset import get_dataloader, NotebookDataset
from library.model import DCAN
from library.train import train_model
from library.inference import predict


def run_demo():
    print("=== Starting AI4Code Solution Demo ===")

    # 1. Configuration & Setup
    # We override Config values to run a fast demo on a small subset of data.
    print("\n[1] Configuring environment for demo...")

    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 notebooks for speed
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Use 0 workers for simple debugging/compatibility

    # Redirect working directory to a demo folder
    Config.WORKING_DIR = "./working/demo_run"

    # Manually update dependent paths since they were defined at class level
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Run setup to create directories
    Config.setup()
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Feature Precomputation
    print("\n[2] Precomputing features (Simulating Preprocessing)...")
    # This will use the SentenceTransformer to encode the text of the 50 sampled notebooks
    # We force reload to ensure we generate the small debug dataset
    precompute_features(load_cached_data=False)

    # Verification
    assert os.path.exists(Config.TRAIN_FEATURES_PATH), "Train features parquet missing"
    assert os.path.exists(Config.VAL_FEATURES_PATH), "Val features parquet missing"
    assert os.path.exists(Config.TEST_FEATURES_PATH), "Test features parquet missing"

    df_check = pd.read_parquet(Config.TRAIN_FEATURES_PATH)
    print(f"Train features shape: {df_check.shape}")
    assert (
        len(df_check) <= Config.DEBUG_SAMPLE_SIZE
    ), "Train set size exceeds debug limit"
    required_cols = ["id", "code_embeddings", "markdown_embeddings", "markdown_labels"]
    for col in required_cols:
        assert col in df_check.columns, f"Missing column {col} in features"

    # 3. Dataset & DataLoader
    print("\n[3] Verifying Dataset and DataLoader...")
    train_loader = get_dataloader(
        split="train", batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify Batch Structure
    print("Batch keys:", batch.keys())
    assert "code_features" in batch
    assert "markdown_features" in batch
    assert "labels" in batch
    assert "code_mask" in batch

    # Verify Shapes
    # code_features: (B, Max_Code_Len, Hidden)
    # markdown_features: (B, Max_Md_Len, Hidden)
    B = batch["code_features"].size(0)
    assert B == Config.BATCH_SIZE or B == len(df_check), f"Batch size mismatch: {B}"
    assert batch["code_features"].size(2) == Config.INPUT_DIM
    assert batch["markdown_features"].size(2) == Config.INPUT_DIM

    # Verify Labels
    # Labels should be indices of code cells (or sink). Max label <= Max Code Len
    labels = batch["labels"]
    # Filter out padding (-100)
    valid_labels = labels[labels != -100]
    if len(valid_labels) > 0:
        max_label = valid_labels.max().item()
        # The label is the index of the code cell following the md cell.
        # It can be at most the number of code cells (which implies 'end of notebook').
        # We just check it's non-negative.
        assert max_label >= 0
        print(f"Max label in batch: {max_label}")

    # 4. Model Forward Pass
    print("\n[4] Verifying Model Architecture...")
    device = Config.DEVICE
    model = DCAN().to(device)

    # Move batch to device
    code_feat = batch["code_features"].to(device)
    code_mask = batch["code_mask"].to(device)
    md_feat = batch["markdown_features"].to(device)
    md_mask = batch["markdown_mask"].to(device)

    logits = model(code_feat, code_mask, md_feat, md_mask)

    print(f"Logits shape: {logits.shape}")
    # Expected: (B, Num_MD, Num_Code + 1)
    # Num_Code + 1 because of the sink token
    assert logits.dim() == 3
    assert logits.size(0) == B
    assert logits.size(1) == md_feat.size(1)
    assert logits.size(2) == code_feat.size(1) + 1

    print("Model forward pass successful.")

    # 5. Training Loop
    print("\n[5] Running Training Loop (1 Epoch)...")
    # This calls the provided training function
    train_model(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE)

    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model file was not saved after training"

    # 6. Inference
    print("\n[6] Running Inference...")
    predict(batch_size=Config.BATCH_SIZE)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file missing"
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    assert "id" in df_sub.columns and "cell_order" in df_sub.columns
    # Check if we have rows (limited by debug sample size of test set)
    assert len(df_sub) > 0

    # 7. Metric Verification
    print("\n[7] Verifying Metric Calculation (Kendall Tau)...")
    # Perfect match
    gt1 = ["a", "b", "c", "d"]
    pred1 = ["a", "b", "c", "d"]
    score1 = compute_kendall_tau([pred1], [gt1])
    print(f"Perfect match score: {score1}")
    assert abs(score1 - 1.0) < 1e-6

    # Worst case (Reverse)
    # For n=4, pairs = 4*3/2 = 6.
    # Reverse order swaps = 6.
    # K = 1 - 4 * (6/6) = -3.0? No, formula is 1 - 4 * (S / (n(n-1)))
    # Wait, the formula in description is: K = 1 - 4 * Sum(S) / Sum(n(n-1))
    # Standard Kendall Tau is usually 1 - 2*S / (n(n-1)/2) = 1 - 4*S / (n(n-1)). Correct.
    # So for reverse: S = n(n-1)/2.
    # K = 1 - 4 * (n(n-1)/2) / (n(n-1)) = 1 - 2 = -1. Correct.
    gt2 = ["a", "b", "c", "d"]
    pred2 = ["d", "c", "b", "a"]
    score2 = compute_kendall_tau([pred2], [gt2])
    print(f"Reverse match score: {score2}")
    assert abs(score2 - (-1.0)) < 1e-6

    # One swap
    # n=3, total pairs = 3*2 = 6 (denominator term n(n-1))
    # gt: a b c
    # pred: a c b (1 swap needed)
    # K = 1 - 4 * (1 / 6) = 1 - 0.666 = 0.333
    gt3 = ["a", "b", "c"]
    pred3 = ["a", "c", "b"]
    score3 = compute_kendall_tau([pred3], [gt3])
    print(f"One swap score (n=3): {score3}")
    assert abs(score3 - (1.0 - 4 / 6.0)) < 1e-6

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
