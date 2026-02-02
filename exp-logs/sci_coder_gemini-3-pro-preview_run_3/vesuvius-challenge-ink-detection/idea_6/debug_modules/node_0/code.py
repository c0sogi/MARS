import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, rle_encoding, fbeta_score
from library.model import SGDN
from library.losses import BCEDiceLoss
from library.data import InkDataset
from library.train import train_one_epoch, validate
from library.inference import predict_fragment


def run_demo():
    print("=== Starting Vesuvius Ink Detection Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Isolation
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Redirect output paths to a demo-specific directory
    Config.WORKING_DIR = Path("./working/demo_execution")
    Config.CACHE_DIR = Path("./working/demo_cache")
    Config.SUBMISSION_PATH = Config.WORKING_DIR / "submission.csv"

    # Enable Debug mode to use small subsets of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small sample for demonstration

    # Training Hyperparameters for quick execution
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Ensure directories exist
    if Config.WORKING_DIR.exists():
        shutil.rmtree(Config.WORKING_DIR)
    Config.WORKING_DIR.mkdir(parents=True, exist_ok=True)
    Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Working Dir: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test RLE Encoding
    # Mask: 0 1 1 0 0 1 0
    # 1-based indices: 2, 3 (len 2) and 6 (len 1) -> "2 2 6 1"
    dummy_mask = np.array([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=np.uint8)
    # Flattened: 0 1 1 0 0 1 0 0
    # Indices:   1 2 3 4 5 6 7 8
    # Runs: start at 2 len 2, start at 6 len 1
    expected_rle = "2 2 6 1"
    calculated_rle = rle_encoding(dummy_mask)
    assert (
        calculated_rle == expected_rle
    ), f"RLE failed. Got {calculated_rle}, expected {expected_rle}"
    print("    RLE Encoding: OK")

    # Test F-beta Score
    # Pred: 1 1 0 0, Target: 1 0 1 0
    # TP=1, FP=1, FN=1. Beta=0.5
    # p = 1/2, r = 1/2. F0.5 = (1.25 * 0.25) / (0.25 * 0.5 + 0.5) = 0.3125 / 0.625 = 0.5
    dummy_preds = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])  # Logits
    dummy_targets = torch.tensor([[[[1, 0], [1, 0]]]])
    score = fbeta_score(dummy_preds, dummy_targets, beta=0.5, threshold=0.0)
    assert 0.0 <= score <= 1.0, "F-beta score out of range"
    print(f"    F-beta Score check: {score:.4f} (OK)")

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture (SGDN)
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture (SGDN)...")
    model = SGDN().to(device)

    # Create dummy input: (Batch, Z_DIM, H, W)
    dummy_input = torch.randn(2, Config.Z_DIM, Config.PATCH_SIZE, Config.PATCH_SIZE).to(
        device
    )

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (Batch, 1, H, W)
    expected_shape = (2, 1, Config.PATCH_SIZE, Config.PATCH_SIZE)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Got {output.shape}, expected {expected_shape}"
    print(f"    Forward pass successful. Output shape: {output.shape}")

    # -------------------------------------------------------------------------
    # 4. Verify Loss Function
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Loss Function (BCEDiceLoss)...")
    criterion = BCEDiceLoss()
    dummy_logits = torch.randn(2, 1, 64, 64).to(device)
    dummy_labels = torch.randint(0, 2, (2, 1, 64, 64)).float().to(device)

    loss = criterion(dummy_logits, dummy_labels)
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"
    print(f"    Loss calculation successful. Value: {loss.item():.4f}")

    # -------------------------------------------------------------------------
    # 5. Data Loading & Training Simulation
    # -------------------------------------------------------------------------
    print("\n[5] Simulating Training Loop...")

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    train_ids = df_train["fragment_id"].astype(str).tolist()

    # Initialize Dataset (Train)
    # Note: This might take a few seconds to load/cache the volume
    print(f"    Loading Train Dataset for Fragment IDs: {train_ids}...")
    train_dataset = InkDataset(
        split="train",
        fragment_ids=train_ids,
        samples_per_epoch=Config.DEBUG_SAMPLE_SIZE,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch
    print("    Running training epoch...")
    avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"    Epoch complete. Average Loss: {avg_loss:.4f}")

    # Save this model as "best_model" for the inference step
    model_path = Config.WORKING_DIR / "best_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"    Model saved to {model_path}")

    # Save a dummy threshold file
    with open(Config.WORKING_DIR / "threshold.txt", "w") as f:
        f.write("0.5")

    # -------------------------------------------------------------------------
    # 6. Inference Simulation
    # -------------------------------------------------------------------------
    print("\n[6] Simulating Inference...")

    # Load Test Metadata
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    test_ids = df_test["fragment_id"].astype(str).tolist()

    submission_rows = []

    # Run inference on test fragments
    model.eval()
    for fid in test_ids:
        print(f"    Processing Test Fragment {fid}...")
        # Using the predict_fragment function from library.inference
        # This handles TTA, sliding window, and reconstruction
        rle = predict_fragment(model, fid, device, threshold=0.5)

        # Verify RLE format (simple check)
        if len(rle) > 0:
            parts = rle.split()
            assert (
                len(parts) % 2 == 0
            ), "RLE string must have even number of elements (start length pairs)"
            assert all(
                x.isdigit() for x in parts
            ), "RLE string must contain only digits"

        submission_rows.append({"Id": fid, "Predicted": rle})
        print(f"    Fragment {fid} prediction generated (RLE Length: {len(rle)} chars)")

    # Generate Submission File
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)

    assert Config.SUBMISSION_PATH.exists(), "Submission file was not created"
    print(f"\n[SUCCESS] Demo completed successfully.")
    print(f"Submission saved at: {Config.SUBMISSION_PATH}")
    print(f"Content preview:\n{sub_df.head()}")


if __name__ == "__main__":
    run_demo()
