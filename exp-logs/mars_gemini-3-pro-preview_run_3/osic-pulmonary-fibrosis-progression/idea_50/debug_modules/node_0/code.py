import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_metric, AverageMeter
from library.loss import MetricAlignedLaplaceLoss
from library.data import get_dataloaders
from library.model import CRDSNet
from library.train import train_one_epoch, evaluate


def create_subset_metadata(num_patients=2):
    """
    Creates a small subset of the metadata to ensure the demo runs quickly.
    Updates Config paths to point to these temporary files.
    """
    print(f"Creating metadata subset with {num_patients} patients for speed...")

    # Create temp directory
    temp_dir = "./working/demo_subset"
    os.makedirs(temp_dir, exist_ok=True)

    # Load original metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Sample patients
    train_patients = train_df["Patient"].unique()[:num_patients]
    val_patients = val_df["Patient"].unique()[:num_patients]
    test_patients = test_df["Patient"].unique()[:num_patients]

    # Filter DataFrames
    train_sub = train_df[train_df["Patient"].isin(train_patients)].copy()
    val_sub = val_df[val_df["Patient"].isin(val_patients)].copy()
    test_sub = test_df[test_df["Patient"].isin(test_patients)].copy()

    # Save to temp location
    train_path = os.path.join(temp_dir, "train.csv")
    val_path = os.path.join(temp_dir, "val.csv")
    test_path = os.path.join(temp_dir, "test.csv")

    train_sub.to_csv(train_path, index=False)
    val_sub.to_csv(val_path, index=False)
    test_sub.to_csv(test_path, index=False)

    # Override Config paths
    Config.TRAIN_CSV = train_path
    Config.VAL_CSV = val_path
    Config.TEST_CSV = test_path

    # Also override working dir to avoid messing with real experiment cache
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Re-setup directories
    Config.setup()

    print("Metadata subset created and Config updated.")


def demo_utils():
    print("\n--- Demonstrating Utils ---")

    # Test calculate_metric
    # Scenario: True=2000, Pred=2100, Sigma=100
    # Delta = 100
    # Sigma_clipped = max(100, 70) = 100
    # Metric = - (sqrt(2) * 100) / 100 - ln(sqrt(2) * 100)
    #        = - 1.4142 - ln(141.42)
    #        = - 1.4142 - 4.9517 = -6.3659

    y_true = np.array([2000.0])
    y_pred = np.array([2100.0])
    y_std = np.array([100.0])

    score = calculate_metric(y_true, y_pred, y_std)
    print(f"Metric Score (True=2000, Pred=2100, Sigma=100): {score:.4f}")

    # Expected calculation
    expected = -(np.sqrt(2) * 100) / 100 - np.log(np.sqrt(2) * 100)
    assert np.isclose(score, expected, atol=1e-4), "Metric calculation mismatch!"
    print("Metric calculation verified.")


def demo_data_loading():
    print("\n--- Demonstrating Data Loading ---")

    # Use small batch size for demo
    batch_size = 2
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=0,  # Avoid multiprocessing overhead in demo
        load_cached_data=True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Fetch one batch
    images, tabular, targets = next(iter(train_loader))

    print(f"Batch Shapes:")
    print(f"  Images:  {images.shape}")
    print(f"  Tabular: {tabular.shape}")
    print(f"  Targets: {targets.shape}")

    # Assertions
    assert images.shape == (
        batch_size,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image tensor shape"
    assert tabular.shape == (
        batch_size,
        Config.TABULAR_INPUT_DIM,
    ), "Incorrect tabular tensor shape"
    assert targets.shape == (batch_size,), "Incorrect target shape"

    return train_loader, val_loader, test_loader


def demo_model_and_loss(train_loader):
    print("\n--- Demonstrating Model and Loss ---")

    device = Config.DEVICE
    model = CRDSNet().to(device)
    criterion = MetricAlignedLaplaceLoss()

    # Get a batch
    images, tabular, targets = next(iter(train_loader))
    images = images.to(device)
    tabular = tabular.to(device)
    targets = targets.to(device)

    # Forward Pass
    preds = model(images, tabular)

    print(f"Prediction Shape: {preds.shape}")
    assert preds.shape == (images.size(0), 2), "Prediction shape mismatch"

    mu, sigma = preds[:, 0], preds[:, 1]
    print(f"Predicted Mean (scaled): {mu.detach().cpu().numpy()}")
    print(f"Predicted Sigma (scaled): {sigma.detach().cpu().numpy()}")

    # Assert Sigma Positivity
    assert (sigma > 0).all(), "Model predicted non-positive sigma!"

    # Calculate Loss
    loss = criterion(preds, targets)
    print(f"Loss Value: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    return model, criterion


def demo_training_step(model, train_loader, criterion):
    print("\n--- Demonstrating Training Step ---")

    device = Config.DEVICE
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch (which is just a few batches in our subset)
    # We use the provided train_one_epoch function
    avg_loss = train_one_epoch(
        epoch=0,
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    print(f"Training Step Complete. Average Loss: {avg_loss:.4f}")


def demo_inference(model, test_loader):
    print("\n--- Demonstrating Inference and Submission ---")

    device = Config.DEVICE
    model.eval()

    results = []
    target_mean = Config.TARGET_MEAN
    target_std = Config.TARGET_STD

    print("Running inference on test set...")
    with torch.no_grad():
        for images, tabular, pat_week_ids in test_loader:
            images = images.to(device)
            tabular = tabular.to(device)

            preds = model(images, tabular)

            mu_scaled = preds[:, 0].cpu().numpy()
            sigma_scaled = preds[:, 1].cpu().numpy()

            # Inverse Transform
            mu_pred = mu_scaled * target_std + target_mean
            sigma_pred = sigma_scaled * target_std

            # Clip Sigma
            sigma_pred = np.maximum(sigma_pred, 70)

            for i, pat_week in enumerate(pat_week_ids):
                results.append(
                    {
                        "Patient_Week": pat_week,
                        "FVC": mu_pred[i],
                        "Confidence": sigma_pred[i],
                    }
                )

    # Create DataFrame
    sub_df = pd.DataFrame(results)
    print(f"Generated {len(sub_df)} predictions.")
    print(sub_df.head())

    # Save
    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not created"


if __name__ == "__main__":
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Create Subset for Speed
    create_subset_metadata(num_patients=3)

    # 3. Run Demos
    demo_utils()

    # 4. Data Loading
    train_loader, val_loader, test_loader = demo_data_loading()

    # 5. Model & Loss
    model, criterion = demo_model_and_loss(train_loader)

    # 6. Training Loop
    demo_training_step(model, train_loader, criterion)

    # 7. Inference
    demo_inference(model, test_loader)

    print("\nAll demonstrations completed successfully.")
