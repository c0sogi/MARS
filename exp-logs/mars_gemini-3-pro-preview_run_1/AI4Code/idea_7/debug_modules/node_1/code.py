import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import set_seed, compute_kendall_tau
from library.preprocessor import Preprocessor
from library.dataset import NotebookDataset, custom_collate_fn
from library.model import CorrectedDCAN
from library.engine import train_one_epoch, validate
from library.inference import predict_and_rank, generate_submission_dataframe


def run_demonstration():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup and Configuration Override
    # We modify the Config class directly to adapt it for a quick demo run.
    print("\n[1] Configuring environment for demo...")

    # Use a specific directory for this demo to avoid overwriting real experiment data
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_FEATURES_PATH = os.path.join(DEMO_DIR, "train_features.parquet")
    Config.VAL_FEATURES_PATH = os.path.join(DEMO_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(DEMO_DIR, "test_features.parquet")
    Config.MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Override Hyperparameters for speed
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Only process 20 notebooks
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # 2. Verify Metric Logic
    print("\n[2] Verifying Metric Logic (Kendall Tau)...")
    # Case 1: Perfect match
    df_gt = pd.DataFrame({"id": ["nb1"], "cell_order": ["a b c d"]})
    df_pred_perfect = pd.DataFrame({"id": ["nb1"], "cell_order": ["a b c d"]})
    score_perfect = compute_kendall_tau(df_gt, df_pred_perfect)
    assert np.isclose(score_perfect, 1.0), f"Expected 1.0, got {score_perfect}"

    # Case 2: Complete reversal (worst case)
    df_pred_worst = pd.DataFrame({"id": ["nb1"], "cell_order": ["d c b a"]})
    # n=4, pairs=6. Inversions=6. Score = 1 - 4*(6/6) = -3.0?
    # Formula: 1 - 4 * (S / (n(n-1)))
    # n(n-1) = 12. S=6. 1 - 4*(6/12) = 1 - 2 = -1.0. Correct.
    score_worst = compute_kendall_tau(df_gt, df_pred_worst)
    assert np.isclose(score_worst, -1.0), f"Expected -1.0, got {score_worst}"
    print("    Metric verification passed.")

    # 3. Preprocessing
    print("\n[3] Running Preprocessor...")
    # This will read metadata, load a few notebooks, encode text with MPNet, and save parquet
    preprocessor = Preprocessor()
    # Force process (load_cached_data=False) to ensure we test the logic
    preprocessor.process_all(load_cached_data=False)

    assert os.path.exists(
        Config.TRAIN_FEATURES_PATH
    ), "Train features parquet not created"
    assert os.path.exists(Config.VAL_FEATURES_PATH), "Val features parquet not created"
    assert os.path.exists(
        Config.TEST_FEATURES_PATH
    ), "Test features parquet not created"
    print("    Preprocessing complete. Files generated.")

    # 4. Dataset and DataLoader
    print("\n[4] Testing Dataset and DataLoader...")
    train_ds = NotebookDataset(Config.TRAIN_FEATURES_PATH, is_test=False)
    assert len(train_ds) > 0, "Train dataset is empty"

    # Test collate function with a DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    batch = next(iter(train_loader))
    print(f"    Batch keys: {batch.keys()}")

    # Verify shapes
    # code_emb: (B, Max_Code_Len, 768)
    # md_emb: (B, Max_Md_Len, 768)
    # labels: (B, Max_Md_Len)
    B = batch["code_emb"].shape[0]
    assert B <= Config.BATCH_SIZE
    assert batch["code_emb"].shape[2] == Config.EMBEDDING_DIM
    assert batch["md_emb"].shape[2] == Config.EMBEDDING_DIM
    assert batch["labels"].shape == batch["md_mask"].shape

    print("    DataLoader shapes verified.")

    # 5. Model Initialization and Forward Pass
    print("\n[5] Testing Model...")
    model = CorrectedDCAN().to(Config.DEVICE)

    # Move batch to device
    code_emb = batch["code_emb"].to(Config.DEVICE)
    md_emb = batch["md_emb"].to(Config.DEVICE)
    code_mask = batch["code_mask"].to(Config.DEVICE)
    md_mask = batch["md_mask"].to(Config.DEVICE)
    code_lens = batch["code_lens"].to(Config.DEVICE)

    logits = model(code_emb, md_emb, code_mask, md_mask, code_lens)

    # Logits shape should be (B, M, L+1)
    # L is the sequence length of code_emb in this batch
    L = code_emb.shape[1]
    M = md_emb.shape[1]

    assert logits.shape == (
        B,
        M,
        L + 1,
    ), f"Expected shape {(B, M, L + 1)}, got {logits.shape}"
    print("    Model forward pass successful.")

    # 6. Training Loop (One Epoch)
    print("\n[6] Testing Training Step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
    print(f"    Training step complete. Loss: {loss:.4f}")
    assert isinstance(loss, float)
    assert loss > 0

    # 7. Validation Step
    print("\n[7] Testing Validation Step...")
    val_ds = NotebookDataset(Config.VAL_FEATURES_PATH, is_test=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, collate_fn=custom_collate_fn
    )
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH).head(Config.DEBUG_SUBSET_SIZE)

    val_score = validate(model, val_loader, df_val_meta, Config.DEVICE)
    print(f"    Validation complete. Kendall Tau: {val_score:.4f}")
    assert -1.0 <= val_score <= 1.0

    # 8. Inference
    print("\n[8] Testing Inference...")
    test_ds = NotebookDataset(Config.TEST_FEATURES_PATH, is_test=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, collate_fn=custom_collate_fn
    )

    # Run prediction logic
    df_scores = predict_and_rank(model, test_loader, Config.DEVICE)
    assert "id" in df_scores.columns
    assert "rank_score" in df_scores.columns

    # Generate submission
    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH).head(Config.DEBUG_SUBSET_SIZE)
    df_submission = generate_submission_dataframe(df_scores, df_test_meta)

    assert "id" in df_submission.columns
    assert "cell_order" in df_submission.columns
    assert len(df_submission) == len(df_test_meta)

    # Save submission
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
