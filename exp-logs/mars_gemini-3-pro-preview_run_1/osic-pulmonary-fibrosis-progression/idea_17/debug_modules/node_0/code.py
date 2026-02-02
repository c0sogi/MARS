import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import LungDataset
from library.model import DualAxisNet
from library.loss import LaplaceLogLikelihoodLoss
from library.engine import Engine


def main():
    print("=== Starting Library Usage Demo ===")

    # 1. Configure for Speed and Reproducibility
    # We modify the Config state to run a fast debug session
    print("\n[1] Configuring Environment...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Process only 10 patients for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Ensure necessary directories exist (handled by Config.setup(), but good to double check)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("Configuration set to DEBUG mode with reduced sample size.")

    # 2. Dataset Initialization and Verification
    print("\n[2] Initializing Datasets...")
    train_ds = LungDataset(mode="train")
    val_ds = LungDataset(mode="val")
    test_ds = LungDataset(mode="test")

    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Val Dataset Size:   {len(val_ds)}")
    print(f"Test Dataset Size:  {len(test_ds)}")

    # Verify a single sample
    print("Verifying data sample structure...")
    sample = train_ds[0]

    # Check keys
    required_keys = [
        "axial",
        "coronal",
        "tabular",
        "skip",
        "meta",
        "target",
        "patient_week",
    ]
    for k in required_keys:
        assert k in sample, f"Sample missing key: {k}"

    # Check shapes
    # Images: (3, 224, 224)
    assert sample["axial"].shape == (
        3,
        224,
        224,
    ), f"Axial shape mismatch: {sample['axial'].shape}"
    assert sample["coronal"].shape == (
        3,
        224,
        224,
    ), f"Coronal shape mismatch: {sample['coronal'].shape}"
    # Tabular: (7,) -> Age, Pct, Sex(2), Smoke(3)
    assert sample["tabular"].shape == (
        7,
    ), f"Tabular shape mismatch: {sample['tabular'].shape}"
    # Skip: (2,) -> Base_FVC, Pct
    assert sample["skip"].shape == (2,), f"Skip shape mismatch: {sample['skip'].shape}"

    print("Data sample verification passed.")

    # 3. Model Initialization and Forward Pass
    print("\n[3] Initializing Model and Running Forward Pass...")
    device = Config.DEVICE
    model = DualAxisNet().to(device)

    # Create a batch of size 2 by repeating the sample
    batch = {
        "axial": torch.stack([sample["axial"], sample["axial"]]).to(device),
        "coronal": torch.stack([sample["coronal"], sample["coronal"]]).to(device),
        "tabular": torch.stack([sample["tabular"], sample["tabular"]]).to(device),
        "skip": torch.stack([sample["skip"], sample["skip"]]).to(device),
        "meta": torch.stack([sample["meta"], sample["meta"]]).to(device),
        "target": torch.tensor([sample["target"], sample["target"]]).to(device),
        "patient_week": [sample["patient_week"], sample["patient_week"]],
    }

    model.eval()
    with torch.no_grad():
        outputs = model(batch)

    # Output should be (Batch, 3) -> [alpha, sigma_base, sigma_growth]
    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (2, 3), "Model output shape incorrect."
    print("Forward pass successful.")

    # 4. Loss Function Verification
    print("\n[4] Verifying Loss Function...")
    criterion = LaplaceLogLikelihoodLoss()

    # Calculate loss
    loss = criterion(outputs, batch["target"], batch["meta"])
    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss returned NaN."
    assert loss.dim() == 0, "Loss should be a scalar."
    print("Loss function verification passed.")

    # 5. Training Loop Demonstration
    print("\n[5] Running Training Loop (1 Epoch)...")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Setup Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Engine
    engine = Engine(model, optimizer, device=device)

    # Run Fit
    # This will train for 1 epoch, evaluate on val set, and save checkpoint
    engine.fit(train_loader, val_loader, criterion, epochs=Config.EPOCHS)

    # Verify Checkpoint
    checkpoint_path = os.path.join(Config.WORKING_DIR, "checkpoints", "best_model.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print("Training loop completed and checkpoint saved.")

    # 6. Inference Demonstration
    print("\n[6] Running Inference on Test Set...")

    # For demo speed, we limit the test loader to a small batch if possible,
    # but since Dataset structure for test is fixed, we just run it.
    # The test set is ~1900 rows, inference is fast.
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    engine.predict(test_loader)

    # Verify Submission
    submission_path = Config.SUBMISSION_FILE
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Check columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in expected_cols:
        assert col in df_sub.columns, f"Submission missing column: {col}"

    # Check logic (Confidence clipped at 70)
    min_conf = df_sub["Confidence"].min()
    assert min_conf >= 70, f"Found confidence value {min_conf} < 70, clipping failed."

    print("Inference and submission generation successful.")
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
