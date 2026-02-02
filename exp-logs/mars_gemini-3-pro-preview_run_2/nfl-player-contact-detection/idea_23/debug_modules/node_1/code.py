import os
import pandas as pd
import numpy as np
import torch
import shutil
from library.config import Config
from library.utils import seed_everything, optimize_threshold
from library.feature_engineering import FeatureEngineer
from library.data_loader import get_dataloaders, get_test_loader
from library.model import KCVRNet
from library.loss import FocalLoss
from library.trainer import Trainer


def create_demo_metadata():
    """
    Creates a small subset of the metadata for demonstration purposes
    to ensure the script runs quickly.
    """
    print("Creating demo metadata subsets...")

    # Load original metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Sample 2 unique game_plays for training/validation
    # We pick ones that definitely exist in tracking data
    unique_gps = train_meta["game_play"].unique()
    sampled_gps = unique_gps[:2]  # Take first 2

    demo_train_meta = train_meta[train_meta["game_play"] == sampled_gps[0]].copy()
    demo_val_meta = train_meta[train_meta["game_play"] == sampled_gps[1]].copy()

    # Sample 1 unique game_play for testing
    test_gps = test_meta["game_play"].unique()
    demo_test_meta = test_meta[test_meta["game_play"] == test_gps[0]].copy()

    # Define paths
    demo_dir = "./working/demo_data"
    os.makedirs(demo_dir, exist_ok=True)

    train_path = os.path.join(demo_dir, "train_meta.csv")
    val_path = os.path.join(demo_dir, "val_meta.csv")
    test_path = os.path.join(demo_dir, "test_meta.csv")

    # Save
    demo_train_meta.to_csv(train_path, index=False)
    demo_val_meta.to_csv(val_path, index=False)
    demo_test_meta.to_csv(test_path, index=False)

    print(f"Demo data saved to {demo_dir}")
    return train_path, val_path, test_path


def run_demo():
    # 1. Setup
    seed_everything(42)

    # 2. Prepare Data
    train_path, val_path, test_path = create_demo_metadata()

    # 3. Override Config for Speed and Demo Isolation
    print("Configuring environment...")
    Config.TRAIN_METADATA_PATH = train_path
    Config.VAL_METADATA_PATH = val_path
    Config.TEST_METADATA_PATH = test_path

    # Use a specific working dir for this run to verify file creation
    Config.WORKING_DIR = "./working/demo_run_v1"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Reduce compute load
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 256
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # 4. Feature Engineering & Data Loading
    print("Initializing DataLoaders (runs Feature Engineering)...")
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Verify DataLoaders
    X_kin_batch, X_vis_batch, y_batch = next(iter(train_loader))

    print(
        f"Batch Shapes - Kinematic: {X_kin_batch.shape}, Visual: {X_vis_batch.shape}, Target: {y_batch.shape}"
    )

    # Assertions
    assert X_kin_batch.dim() == 2, "Kinematic features should be 2D (Batch, Features)"
    assert X_vis_batch.dim() == 2, "Visual features should be 2D (Batch, Features)"
    assert y_batch.dim() == 2, "Targets should be 2D (Batch, 1)"
    assert not torch.isnan(X_kin_batch).any(), "Kinematic features contain NaNs"
    assert not torch.isnan(X_vis_batch).any(), "Visual features contain NaNs"

    # 5. Model Initialization
    print("Initializing KCVRNet Model...")
    model = KCVRNet()

    # Verify Forward Pass
    with torch.no_grad():
        model.eval()
        logits = model(X_kin_batch, X_vis_batch)

    assert logits.shape == (
        X_kin_batch.size(0),
        1,
    ), f"Output shape mismatch. Expected ({X_kin_batch.size(0)}, 1), got {logits.shape}"
    print("Model forward pass successful.")

    # 6. Loss Function Verification
    print("Verifying Focal Loss...")
    criterion = FocalLoss()
    loss = criterion(logits, y_batch)

    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    print(f"Initial Loss: {loss.item():.4f}")

    # 7. Training Loop
    print("Starting Training...")
    trainer = Trainer(model, device=Config.DEVICE)

    # Run fit
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Verify model was saved
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "Best model file was not saved."
    print("Training complete. Model saved.")

    # 8. Inference & Threshold Optimization
    print("Running Inference on Test Set...")
    test_loader, test_ids = get_test_loader(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    all_probs = []
    with torch.no_grad():
        for X_kin, X_vis in test_loader:
            X_kin = X_kin.to(Config.DEVICE)
            X_vis = X_vis.to(Config.DEVICE)
            logits = model(X_kin, X_vis)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs).flatten()

    assert len(all_probs) == len(
        test_ids
    ), "Number of predictions does not match number of test IDs"

    # Create submission dataframe
    submission = pd.DataFrame(
        {
            "contact_id": test_ids,
            "contact": (all_probs > 0.5).astype(int),  # Simple threshold for demo
        }
    )

    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Demonstrate optimize_threshold with dummy ground truth for test
    # (In real scenario, we use validation set for this, which Trainer.validate already does)
    print("Demonstrating Threshold Optimization (using Validation data)...")

    # Get validation predictions
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for X_kin, X_vis, y in val_loader:
            X_kin = X_kin.to(Config.DEVICE)
            X_vis = X_vis.to(Config.DEVICE)
            logits = model(X_kin, X_vis)
            val_probs.append(torch.sigmoid(logits).cpu().numpy())
            val_targets.append(y.cpu().numpy())

    val_probs = np.concatenate(val_probs)
    val_targets = np.concatenate(val_targets)

    best_thresh, best_mcc = optimize_threshold(val_targets, val_probs)
    print(
        f"Optimization Result - Best Threshold: {best_thresh:.4f}, Best MCC: {best_mcc:.4f}"
    )

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
