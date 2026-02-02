import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, alaska_weighted_auc
from library.dataset import StegoDataset
from library.model import DRRENet
from library.engine import StegoEngine


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("--- Setting up Configuration for Demo ---")

    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 24  # Small subset for speed
    Config.NUM_EPOCHS = 1  # Single epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.WORK_DIR = "./working/demo_run"
    Config.BEST_MODEL_PATH = os.path.join(Config.WORK_DIR, "best_model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORK_DIR, "submission_demo.csv")

    # Re-run setup to ensure new directories exist
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Dataset & DataLoader Demonstration
    # ==========================================
    print("\n--- Initializing Datasets ---")

    # Initialize Datasets
    train_dataset = StegoDataset(split="train")
    val_dataset = StegoDataset(split="val")
    test_dataset = StegoDataset(split="test")

    # Verify Dataset Lengths (should match DEBUG_SAMPLE_SIZE or available data)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    # Assertion to ensure debug mode worked
    assert (
        len(train_dataset) <= Config.DEBUG_SAMPLE_SIZE
    ), "Train dataset size exceeds debug limit"

    # Verify Data Shapes
    sample_img, sample_label = train_dataset[0]
    print(f"Sample Image Shape: {sample_img.shape}")
    print(f"Sample Label: {sample_label}")

    assert sample_img.shape == (
        3,
        512,
        512,
    ), f"Unexpected image shape: {sample_img.shape}"
    assert isinstance(sample_label, torch.Tensor), "Label should be a tensor"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ==========================================
    # 3. Model Demonstration
    # ==========================================
    print("\n--- Initializing Model ---")

    model = DRRENet()
    model.to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, 512, 512).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    # Verify SRM weights are frozen
    assert model.srm.weight.requires_grad is False, "SRM weights should be frozen"

    # ==========================================
    # 4. Metric Logic Verification
    # ==========================================
    print("\n--- Verifying Metric Logic ---")

    # Test Case 1: Perfect Predictions
    y_true = np.array([0, 0, 1, 1])
    y_pred_perfect = np.array([0.1, 0.2, 0.9, 0.8])
    auc_perfect = alaska_weighted_auc(y_true, y_pred_perfect)
    print(f"Perfect Prediction AUC: {auc_perfect}")
    assert np.isclose(auc_perfect, 1.0), "Perfect predictions should yield AUC 1.0"

    # Test Case 2: Random/Bad Predictions
    y_pred_bad = np.array([0.9, 0.8, 0.1, 0.2])  # Inverted
    auc_bad = alaska_weighted_auc(y_true, y_pred_bad)
    print(f"Inverted Prediction AUC: {auc_bad}")
    assert auc_bad < 0.5, "Inverted predictions should yield low AUC"

    # ==========================================
    # 5. Engine & Training Loop Demonstration
    # ==========================================
    print("\n--- Starting Training Loop ---")

    # Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    # Initialize Engine
    engine = StegoEngine(model, device, optimizer, scheduler)

    # Run Training
    # This will run for 1 epoch on the small subset
    engine.train_model(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    # Verify Model Checkpoint
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Checkpoint successfully saved at: {Config.BEST_MODEL_PATH}")
    else:
        # If validation didn't improve (possible with random init and tiny data),
        # force save for the sake of the demo's next step
        print(
            "Validation AUC did not improve (expected on tiny random data). Saving current model manually."
        )
        torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n--- Generating Submission ---")

    engine.generate_submission(test_loader)

    # Verify Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created at: {Config.SUBMISSION_PATH}")
        print("First 5 rows:")
        print(df_sub.head())

        assert list(df_sub.columns) == ["Id", "Label"], "Submission columns incorrect"
        assert len(df_sub) == len(test_dataset), "Submission length mismatch"
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
