import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

# Import from the provided library
from library.config import seed_everything, DEVICE, WORKING_DIR, SUBMISSION_PATH
from library.utils import rle_encode, fbeta_score
from library.dataset import InkDataset
from library.model import InkDetectorFCN
from library.engine import train_model
from library.inference import generate_submission


def run_demonstration():
    # 1. Setup and Reproducibility
    print("--- 1. Setup ---")
    seed_everything(42)
    print(f"Device: {DEVICE}")
    print(f"Working Directory: {WORKING_DIR}")

    # 2. Verify Utilities
    print("\n--- 2. Verifying Utilities ---")
    # Test RLE Encoding
    # Mask:
    # 0 1 1 0
    # 1 1 0 0
    # Flattened: 0 1 1 0 1 1 0 0
    # 1-based indices of '1's: 2, 3, 5, 6
    # Runs: Start 2 Len 2, Start 5 Len 2 -> "2 2 5 2"
    dummy_mask = np.array([[0, 1, 1, 0], [1, 1, 0, 0]], dtype=np.uint8)
    encoded = rle_encode(dummy_mask)
    expected_rle = "2 2 5 2"
    assert (
        encoded == expected_rle
    ), f"RLE Encoding failed. Got {encoded}, expected {expected_rle}"
    print("RLE Encoding: OK")

    # Test F-beta Score
    # Preds: 0.1, 0.9, 0.8, 0.2 -> Binary (thresh 0.5): 0, 1, 1, 0
    # Targets: 0, 1, 1, 0
    # Perfect match -> Score should be 1.0
    dummy_preds = torch.tensor([0.1, 0.9, 0.8, 0.2])
    dummy_targets = torch.tensor([0, 1, 1, 0])
    score = fbeta_score(dummy_preds, dummy_targets, beta=0.5)
    assert np.isclose(score, 1.0), f"F-beta Score failed. Got {score}"
    print("F-beta Score: OK")

    # 3. Verify Dataset Loading
    print("\n--- 3. Initializing Datasets ---")
    print("Note: This may take a moment to load and cache the 3D volumes...")

    # We use a very small number of patches per epoch to ensure the training step is fast
    train_dataset = InkDataset(
        split="train", patches_per_epoch=4, load_cached_data=True
    )
    val_dataset = InkDataset(split="val", patches_per_epoch=2, load_cached_data=True)

    # Verify item retrieval
    vol_patch, label_patch = train_dataset[0]
    print(f"Train Patch Shape: {vol_patch.shape}")
    print(f"Label Patch Shape: {label_patch.shape}")

    # Assertions for shape (Z_DIM=65, PATCH_SIZE=256 defined in config)
    assert vol_patch.shape == (65, 256, 256), "Incorrect volume patch shape"
    assert label_patch.shape == (1, 256, 256), "Incorrect label patch shape"
    print("Dataset Initialization: OK")

    # 4. Verify Model
    print("\n--- 4. Initializing Model ---")
    model = InkDetectorFCN().to(DEVICE)

    # Test forward pass with dummy batch
    dummy_input = torch.randn(2, 65, 256, 256).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1, 256, 256), "Model output shape mismatch"
    print("Model Forward Pass: OK")

    # 5. Run Training Loop
    print("\n--- 5. Running Training Loop (1 Epoch) ---")
    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=0, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=0, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    save_path = WORKING_DIR / "demo_model.pth"

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=1,  # Single epoch for demonstration
        patience=1,
        save_path=save_path,
    )

    assert save_path.exists(), "Model file was not saved after training"
    print("Training Loop: OK")

    # 6. Run Inference
    print("\n--- 6. Running Inference on Test Set ---")
    # generate_submission loads metadata/test.csv, runs inference, and creates submission.csv
    generate_submission(model_path=save_path)

    # Verify submission file
    if SUBMISSION_PATH.exists():
        print(f"Submission file found at {SUBMISSION_PATH}")
        with open(SUBMISSION_PATH, "r") as f:
            lines = f.readlines()
            print(f"Number of lines in submission: {len(lines)}")
            # Header + 1 row for fragment 'a'
            assert len(lines) >= 2, "Submission file is empty or missing rows"
            assert lines[0].strip() == "Id,Predicted", "Submission header is incorrect"
    else:
        raise FileNotFoundError(f"Submission file not found at {SUBMISSION_PATH}")

    print("Inference Pipeline: OK")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demonstration()
