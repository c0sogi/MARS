import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config, setup_directories
from library.utils import seed_everything
from library.data_processing import load_and_process_data, IcebergDataset
from library.model import TripleStreamWideBodyNetwork
from library.training import Trainer


def run_demo():
    print("=== TS-WBN Pipeline Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")
    # Enable Debug mode to use a tiny subset of data (20 samples)
    Config.DEBUG = True
    Config.DEBUG_SIZE = 20

    # Reduce training duration
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4

    # Disable multiprocessing for simple demo execution
    Config.NUM_WORKERS = 0

    # Set a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_PATH = os.path.join(Config.WORKING_DIR, "cache", "processed_data.npz")

    # Initialize directories and seeds
    setup_directories()
    seed_everything(42)
    print("    Configuration updated: DEBUG=True, EPOCHS=2, BATCH_SIZE=4")

    # ---------------------------------------------------------
    # 2. Data Loading and Processing
    # ---------------------------------------------------------
    print("\n[2] Loading and processing data...")
    # We set load_cached_data=False to demonstrate the raw processing logic
    train_data, val_data, test_data, global_stats = load_and_process_data(
        load_cached_data=False
    )

    # Validate Data Shapes
    print("    Verifying data shapes...")
    assert len(train_data["images"]) == Config.DEBUG_SIZE, "Train data size mismatch"
    assert train_data["images"].shape == (
        Config.DEBUG_SIZE,
        75,
        75,
        3,
    ), "Train image shape mismatch"
    assert train_data["angles"].shape == (
        Config.DEBUG_SIZE,
    ), "Train angle shape mismatch"
    assert (
        "min" in global_stats and "max" in global_stats
    ), "Global stats missing min/max"
    print("    Data integrity checks passed.")

    # ---------------------------------------------------------
    # 3. Dataset and DataLoader Creation
    # ---------------------------------------------------------
    print("\n[3] Creating DataLoaders...")

    # Train Dataset (with augmentation)
    train_dataset = IcebergDataset(
        images=train_data["images"],
        angles=train_data["angles"],
        labels=train_data["labels"],
        stats=global_stats,
        transform=True,
    )

    # Validation Dataset (no augmentation)
    val_dataset = IcebergDataset(
        images=val_data["images"],
        angles=val_data["angles"],
        labels=val_data["labels"],
        stats=global_stats,
        transform=False,
    )

    # Test Dataset (no labels)
    test_dataset = IcebergDataset(
        images=test_data["images"],
        angles=test_data["angles"],
        labels=None,
        stats=global_stats,
        transform=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    print(f"    DataLoaders created. Train batches: {len(train_loader)}")

    # ---------------------------------------------------------
    # 4. Model Instantiation and Verification
    # ---------------------------------------------------------
    print("\n[4] Instantiating TripleStreamWideBodyNetwork...")
    model = TripleStreamWideBodyNetwork()
    model.to(Config.DEVICE)

    # Perform a dummy forward pass to verify architecture
    dummy_images = torch.randn(Config.BATCH_SIZE, 3, 75, 75).to(Config.DEVICE)
    dummy_angles = torch.randn(Config.BATCH_SIZE).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_images, dummy_angles)

    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, 1), got {output.shape}"
    print("    Forward pass verification successful.")

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print("\n[5] Starting Training Loop...")
    trainer = Trainer(model, train_loader, val_loader, device=Config.DEVICE)

    # Path to save the best model from this run
    model_save_path = os.path.join(Config.WORKING_DIR, "model_fold_0.pth")

    # Run training
    best_val_loss = trainer.fit(save_path=model_save_path)

    # Verify training results
    assert isinstance(best_val_loss, float), "Trainer did not return a float loss value"
    assert os.path.exists(model_save_path), "Best model file was not saved"
    print(f"    Training completed. Best Validation Loss: {best_val_loss:.6f}")

    # ---------------------------------------------------------
    # 6. Inference and Submission Generation
    # ---------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, angles, _ in test_loader:
            images = images.to(Config.DEVICE)
            angles = angles.to(Config.DEVICE)

            # Forward pass
            logits = model(images, angles)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            predictions.extend(probs.cpu().numpy().flatten())

    assert len(predictions) == len(
        test_data["ids"]
    ), "Number of predictions does not match number of test IDs"

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_data["ids"], "is_iceberg": predictions})

    # Save Submission
    submission_dir = os.path.join(Config.WORKING_DIR, "submission")
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"    Submission saved to {submission_path}")
    print("\nSample Predictions:")
    print(submission_df.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
