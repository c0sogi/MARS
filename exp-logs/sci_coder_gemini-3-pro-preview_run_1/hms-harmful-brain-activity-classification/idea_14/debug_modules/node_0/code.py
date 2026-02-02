import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, kl_divergence
from library.dataset import EEGSeizureDataset
from library.models import CyclicFusionNet
from library.engine import train_one_epoch, validate, inference


def run_demonstration():
    print("--- Starting Demonstration of Harmful Brain Activity Detection Library ---")

    # 1. Setup and Configuration Overrides for Speed
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Override Config for the demo to run fast
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid overhead for small demo
    Config.DEBUG = True

    # define paths
    train_csv_path = Config.TRAIN_CSV
    test_csv_path = Config.TEST_CSV

    # 2. Data Loading & Dataset Verification
    print("\n--- 1. Data Loading & Dataset Verification ---")

    # Load metadata
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Metadata file not found: {train_csv_path}")

    df_train_full = pd.read_csv(train_csv_path)
    print(f"Original Train Metadata: {len(df_train_full)} rows")

    # Subsample for demonstration (20 samples)
    df_demo = df_train_full.head(20).copy()
    print(f"Demo Subset: {len(df_demo)} rows")

    # Instantiate Dataset
    train_dataset = EEGSeizureDataset(df_demo, mode="train", augment=True)

    # Verify __getitem__
    eeg, spec, target = train_dataset[0]

    print(f"EEG Tensor Shape: {eeg.shape}")
    print(f"Spec Tensor Shape: {spec.shape}")
    print(f"Target Shape: {target.shape}")

    # Assertions
    # EEG: (Channels=20, Time=5000)
    assert eeg.shape == (20, 5000), f"Expected EEG shape (20, 5000), got {eeg.shape}"
    # Spec: (Channels=5, H=512, W=512)
    assert spec.shape == (
        5,
        512,
        512,
    ), f"Expected Spec shape (5, 512, 512), got {spec.shape}"
    # Target: (Classes=6)
    assert target.shape == (6,), f"Expected Target shape (6,), got {target.shape}"
    assert torch.is_floating_point(eeg), "EEG data should be float"

    print("Dataset verification passed.")

    # 3. Model Initialization & Forward Pass
    print("\n--- 2. Model Initialization & Forward Pass ---")

    model = CyclicFusionNet()
    model.to(device)

    # Create a dummy batch
    loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    eeg_batch, spec_batch, target_batch = next(iter(loader))

    eeg_batch = eeg_batch.to(device)
    spec_batch = spec_batch.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(eeg_batch, spec_batch)

    print(f"Output Batch Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        6,
    ), f"Expected output (B, 6), got {outputs.shape}"

    # Check Softmax (Sum ~ 1.0)
    sums = outputs.sum(dim=1).cpu().numpy()
    print(f"Probability Sums: {sums}")
    assert np.allclose(
        sums, 1.0, atol=1e-5
    ), "Model outputs do not sum to 1.0 (Softmax failure)"

    print("Model forward pass verification passed.")

    # 4. Training Loop Demonstration
    print("\n--- 3. Training Loop Demonstration ---")

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # Simple scheduler for demo
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Run one epoch using the engine function
    print("Running train_one_epoch...")
    epoch_loss = train_one_epoch(loader, model, optimizer, scheduler, device, epoch=0)

    print(f"Epoch Loss: {epoch_loss:.6f}")
    assert epoch_loss > 0, "Loss should be positive"
    assert not np.isnan(epoch_loss), "Loss should not be NaN"

    print("Training loop verification passed.")

    # 5. Validation Demonstration
    print("\n--- 4. Validation Demonstration ---")

    # Use the same loader as validation for simplicity in this demo
    val_loss = validate(loader, model, device)
    print(f"Validation Loss: {val_loss:.6f}")

    assert val_loss > 0, "Validation loss should be positive"

    print("Validation verification passed.")

    # 6. Metric Utility Verification
    print("\n--- 5. Metric Utility Verification (KL Divergence) ---")

    # Case 1: Identical distributions (Loss should be 0)
    y_true = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]])
    y_pred = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]])
    kl_zero = kl_divergence(y_pred, y_true)
    print(f"KL (Identical): {kl_zero:.6f}")
    assert abs(kl_zero) < 1e-6, "KL Divergence for identical distributions should be ~0"

    # Case 2: Different distributions
    y_pred_diff = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.3, 0.3]])
    kl_diff = kl_divergence(y_pred_diff, y_true)
    print(f"KL (Different): {kl_diff:.6f}")
    assert kl_diff > 0, "KL Divergence should be positive for different distributions"

    print("Metric utility verification passed.")

    # 7. Inference & Submission
    print("\n--- 6. Inference & Submission Generation ---")

    if os.path.exists(test_csv_path):
        df_test_full = pd.read_csv(test_csv_path)
        # Subsample test
        df_test_demo = df_test_full.head(10).copy()
        print(f"Test Demo Subset: {len(df_test_demo)} rows")

        test_dataset = EEGSeizureDataset(df_test_demo, mode="test", augment=False)
        test_loader = DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )

        # Run inference
        preds = inference(test_loader, model, device)
        print(f"Predictions Shape: {preds.shape}")

        assert preds.shape == (len(df_test_demo), 6), "Incorrect prediction shape"

        # Create submission dataframe
        sub_df = pd.DataFrame(preds, columns=Config.TARGET_COLS)
        sub_df["eeg_id"] = df_test_demo["eeg_id"].values

        # Reorder columns
        cols = ["eeg_id"] + Config.TARGET_COLS
        sub_df = sub_df[cols]

        # Normalize
        vote_cols = Config.TARGET_COLS
        sub_df[vote_cols] = sub_df[vote_cols].div(sub_df[vote_cols].sum(axis=1), axis=0)

        # Save
        os.makedirs("./working", exist_ok=True)
        save_path = "./working/submission_demo.csv"
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

        # Verify file content
        saved_df = pd.read_csv(save_path)
        print(f"Loaded Submission Shape: {saved_df.shape}")
        assert saved_df.shape == (
            len(df_test_demo),
            7,
        ), "Saved submission shape mismatch"
        assert list(saved_df.columns) == cols, "Column mismatch in submission"

        print("Inference and submission verification passed.")
    else:
        print("Test metadata not found, skipping inference step.")

    print("\n--- Demonstration Complete ---")


if __name__ == "__main__":
    run_demonstration()
