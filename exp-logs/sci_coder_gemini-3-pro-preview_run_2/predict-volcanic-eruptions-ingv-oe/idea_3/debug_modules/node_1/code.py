import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import provided library modules
from library import config
from library import utils
from library import features
from library import dataset
from library import model
from library import engine


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # 1. Setup
    utils.seed_everything(seed=42)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define paths for demo metadata
    demo_train_path = os.path.join(config.WORKING_DIR, "demo_train.csv")
    demo_val_path = os.path.join(config.WORKING_DIR, "demo_val.csv")

    # 2. Prepare Subset Metadata (for speed)
    print("\n[1/5] Preparing subset metadata...")

    # Load original metadata
    full_train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    full_val_df = pd.read_csv(config.VAL_METADATA_PATH)

    # Sample subsets (20 train, 10 val)
    subset_train = full_train_df.head(20).copy()
    subset_val = full_val_df.head(10).copy()

    # Save subsets
    subset_train.to_csv(demo_train_path, index=False)
    subset_val.to_csv(demo_val_path, index=False)

    print(f"Created demo train set: {len(subset_train)} samples")
    print(f"Created demo val set: {len(subset_val)} samples")

    # 3. Verify Feature Engineering Functions
    print("\n[2/5] Verifying Feature Engineering...")

    # Pick one file to test
    sample_row = subset_train.iloc[0]
    sample_file_path = os.path.join(config.INPUT_DIR, sample_row["file_path"])

    # Load raw data
    df_sensor = pd.read_csv(sample_file_path)

    # Test Spectrogram
    spec = features.get_spectrogram(df_sensor)
    print(f"Spectrogram Shape: {spec.shape}")

    # Assertions for Spectrogram
    # Shape should be (Channels=10, N_MELS=128, Time=~118)
    # Time dim depends on signal length (60001) / hop_length (512) approx 118
    assert spec.dim() == 3, "Spectrogram must be 3D tensor"
    assert spec.shape[0] == 10, "Spectrogram must have 10 channels"
    assert (
        spec.shape[1] == config.N_MELS
    ), f"Spectrogram must have {config.N_MELS} Mel bins"

    # Test Statistics
    stats = features.get_statistics(df_sensor)
    print(f"Number of statistical features extracted: {len(stats)}")
    assert isinstance(stats, dict), "get_statistics should return a dictionary"
    assert len(stats) > 0, "Statistics dictionary is empty"

    # Test SpecAugment
    aug_spec = features.spec_augment(spec.clone())
    assert aug_spec.shape == spec.shape, "Augmentation should not change tensor shape"

    print("Feature verification passed.")

    # 4. Verify Dataset Class
    print("\n[3/5] Verifying Dataset Class...")

    # Initialize Train Dataset (Computes Scalers)
    # Note: This will save scalers to WORKING_DIR based on the subset data
    train_dataset = dataset.VolcanoDataset(
        metadata_path=demo_train_path,
        mode="train",
        augment=True,
        load_cached_stats=False,  # Force recompute for demo
    )

    assert len(train_dataset) == 20, "Dataset length mismatch"

    # Get one item
    spec_t, stats_t, target_t, seg_id = train_dataset[0]

    print(
        f"Dataset Item Shapes -> Spec: {spec_t.shape}, Stats: {stats_t.shape}, Target: {target_t.shape}"
    )

    assert isinstance(spec_t, torch.Tensor)
    assert isinstance(stats_t, torch.Tensor)
    assert isinstance(target_t, torch.Tensor)

    # Initialize Val Dataset (Loads Scalers from Train)
    val_dataset = dataset.VolcanoDataset(
        metadata_path=demo_val_path, mode="val", augment=False, load_cached_stats=False
    )
    assert len(val_dataset) == 10

    print("Dataset verification passed.")

    # 5. Verify Model Architecture
    print("\n[4/5] Verifying Model Architecture...")

    # Determine input dimension for stats branch
    num_stats_features = stats_t.shape[0]

    # Instantiate Model
    net = model.HybridCRNN(num_stats_features=num_stats_features)
    net.to(config.DEVICE)

    # Create dummy batch
    batch_size = 2
    dummy_spec = (
        spec_t.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(config.DEVICE)
    )  # (B, 10, 128, T)
    dummy_stats = (
        stats_t.unsqueeze(0).repeat(batch_size, 1).to(config.DEVICE)
    )  # (B, StatsDim)

    # Forward Pass
    net.eval()
    with torch.no_grad():
        output = net(dummy_spec, dummy_stats)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        batch_size,
    ), f"Expected output shape ({batch_size},), got {output.shape}"

    print("Model verification passed.")

    # 6. Verify Training Loop (Engine)
    print("\n[5/5] Verifying Training Loop...")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True
    )

    # Setup Training Components
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Run Fit (Short duration)
    # We use a temporary save path for the demo model
    demo_model_path = os.path.join(config.WORKING_DIR, "demo_model.pth")

    engine.fit(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=config.DEVICE,
        epochs=2,  # Only 2 epochs for speed
        patience=2,
        save_path=demo_model_path,
        target_mean=train_dataset.target_mean,
        target_std=train_dataset.target_std,
    )

    # Verify model file was created
    assert os.path.exists(demo_model_path), "Model file was not saved after training"
    print(f"Training loop finished. Model saved to {demo_model_path}")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
