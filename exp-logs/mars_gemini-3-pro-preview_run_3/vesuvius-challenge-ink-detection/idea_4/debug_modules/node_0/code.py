import os
import torch
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, rle_encode, fbeta_score
from library.data import get_dataloaders, InkDataset
from library.model import HDNPCA
from library.engine import train_one_epoch, evaluate
from library.inference import predict_full_map, find_best_threshold, generate_submission


def run_demo():
    # 1. Configuration & Setup
    print("--- 1. Configuration & Setup ---")
    # Override Config for a fast demo run
    Config.CACHE_DIR = Path("./working/demo_cache")
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.DEBUG_SAMPLE_SIZE = 20  # Limit dataset size for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure cache dir exists
    if Config.CACHE_DIR.exists():
        shutil.rmtree(Config.CACHE_DIR)
    Config.setup()

    seed_everything(Config.SEED)
    print(f"Cache Directory: {Config.CACHE_DIR}")
    print(f"Device: {Config.DEVICE}")

    # 2. Verify Utility Functions
    print("\n--- 2. Verifying Utility Functions ---")

    # Test RLE Encoding
    # Create a simple 4x4 mask:
    # 0 1 1 0
    # 0 0 0 0
    # 1 0 0 0
    # 0 0 0 0
    # Flattened (row-major): 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
    # Indices (1-based):     1  2  3  4  5  6  7  8  9 10 ...
    # Runs: Start at 2 (len 2), Start at 9 (len 1)
    # Expected RLE: "2 2 9 1"
    dummy_mask = np.zeros((4, 4), dtype=np.uint8)
    dummy_mask[0, 1:3] = 1
    dummy_mask[2, 0] = 1

    rle_out = rle_encode(dummy_mask)
    print(f"RLE Output: {rle_out}")
    assert (
        rle_out == "2 2 9 1"
    ), f"RLE Encoding failed. Expected '2 2 9 1', got '{rle_out}'"

    # Test F-beta Score
    # Preds: 1 1 0 0, Targets: 1 0 0 1
    # TP=1 (idx 0), FP=1 (idx 1), FN=1 (idx 3)
    # Precision = 1/2, Recall = 1/2
    # Beta=0.5 -> Weight precision higher
    preds_t = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    targets_t = torch.tensor([1, 0, 0, 1], dtype=torch.float32)
    score = fbeta_score(preds_t, targets_t, threshold=0.5, beta=0.5)
    print(f"F0.5 Score: {score:.4f}")
    assert 0 <= score <= 1.0, "F-beta score out of range"

    # 3. Data Loading
    print("\n--- 3. Data Loading ---")
    # Use debug=True to trigger the limit_size logic in get_dataloaders
    train_loader, val_loader = get_dataloaders(load_cached=True, debug=True)

    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches: {len(val_loader)}")

    # Verify Train Batch
    train_batch = next(iter(train_loader))
    t_vols, t_lbls = train_batch
    print(f"Train Volume Shape: {t_vols.shape}")  # (B, 65, 256, 256)
    print(f"Train Label Shape: {t_lbls.shape}")  # (B, 1, 256, 256)

    assert t_vols.shape == (
        Config.BATCH_SIZE,
        Config.Z_DIM,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    )
    assert t_lbls.shape == (Config.BATCH_SIZE, 1, Config.PATCH_SIZE, Config.PATCH_SIZE)

    # 4. Model Initialization
    print("\n--- 4. Model Initialization ---")
    model = HDNPCA().to(Config.DEVICE)

    # Forward pass check
    with torch.no_grad():
        dummy_input = t_vols.to(Config.DEVICE)
        dummy_out = model(dummy_input)

    print(f"Model Output Shape: {dummy_out.shape}")
    assert dummy_out.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    )

    # 5. Training Loop Demonstration
    print("\n--- 5. Training Loop Demonstration ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for one epoch on the small subset
    train_loss = train_one_epoch(model, train_loader, optimizer, Config.DEVICE)
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive"

    # Save this model as 'best_model' for inference demo
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"Saved demo model to {Config.BEST_MODEL_PATH}")

    # 6. Evaluation Demonstration
    print("\n--- 6. Evaluation Demonstration ---")
    val_loss, val_f05 = evaluate(model, val_loader, Config.DEVICE)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation F0.5: {val_f05:.4f}")

    # 7. Inference Pipeline Demonstration
    print("\n--- 7. Inference Pipeline Demonstration ---")
    # Instead of running the full run_inference which loads everything,
    # we manually construct the pipeline with a small test subset.

    # A. Load Test Data (Limited)
    test_ds = InkDataset("test", load_cached=True, limit_size=20)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    print(f"Test Dataset Size (Limited): {len(test_ds)}")

    # B. Predict Full Map (using TTA)
    print("Generating probability maps...")
    test_probs = predict_full_map(
        model, test_loader, test_ds, Config.DEVICE, use_tta=True
    )

    # Verify we got a probability map for the test fragment
    test_frag_id = test_ds.fragments[0]["id"]
    assert test_frag_id in test_probs
    print(
        f"Probability map shape for fragment {test_frag_id}: {test_probs[test_frag_id].shape}"
    )

    # C. Threshold Optimization (Mocking with Val data)
    # In a real run, we use val_probs. Here we reuse the logic on the test set
    # just to show the function call, or re-predict on val.
    # Let's re-predict on val quickly for valid threshold finding.
    print("Optimizing threshold on validation subset...")
    val_probs_map = predict_full_map(
        model, val_loader, val_loader.dataset, Config.DEVICE, use_tta=False
    )
    best_th = find_best_threshold(val_probs_map, val_loader.dataset)
    print(f"Selected Threshold: {best_th}")

    # D. Generate Submission
    print("Generating submission file...")
    generate_submission(test_probs, test_ds, best_th, Config.SUBMISSION_PATH)

    assert Config.SUBMISSION_PATH.exists()

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(df_sub.head())
    assert "Id" in df_sub.columns and "Predicted" in df_sub.columns
    assert len(df_sub) > 0

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
