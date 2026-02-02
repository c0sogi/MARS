import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import GNSSSequenceDataset, get_datasets
from library.model import SEResUNet1D
from library.engine import Trainer


def main():
    print("=== GNSS Positioning Pipeline Demo ===")

    # 1. Configuration Setup
    # We modify the default config to run a fast debug session
    config = Config()
    config.DEBUG = True
    config.DEBUG_DRIVE_COUNT = 1  # Process only 1 drive per split for speed
    config.EPOCHS = 1
    config.BATCH_SIZE = 2
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this demo

    print(
        f"Configuration: Device={config.DEVICE}, Debug={config.DEBUG}, Epochs={config.EPOCHS}"
    )

    # 2. Data Preprocessing & Dataset Creation
    print("\n[Data Processing]")
    # load_cached_data=False ensures we run the preprocessing logic from scratch
    train_dataset, val_dataset, test_dataset = get_datasets(
        config, load_cached_data=False
    )

    # Basic validation of datasets
    if train_dataset is None or len(train_dataset) == 0:
        raise RuntimeError(
            "Train dataset is empty. Check input data or debug settings."
        )

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences:   {len(val_dataset) if val_dataset else 0}")
    print(f"Test sequences:  {len(test_dataset) if test_dataset else 0}")

    # 3. Inspect a single sample
    print("\n[Sample Inspection]")
    sample = train_dataset[0]
    print(f"Drive: {sample['drive_id']}, Phone: {sample['phone_name']}")
    print(f"Features shape: {sample['features'].shape}")  # (Channels, Length)
    print(f"Targets shape:  {sample['targets'].shape}")  # (2, Length)

    # Verify feature dimensions match config
    assert (
        sample["features"].shape[0] == config.INPUT_CHANNELS
    ), f"Expected {config.INPUT_CHANNELS} input channels, got {sample['features'].shape[0]}"

    # 4. Create DataLoaders
    print("\n[DataLoader Creation]")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=GNSSSequenceDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=GNSSSequenceDataset.collate_fn,
            num_workers=config.NUM_WORKERS,
        )
        if val_dataset
        else None
    )

    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=GNSSSequenceDataset.collate_fn,
            num_workers=config.NUM_WORKERS,
        )
        if test_dataset
        else None
    )

    # Verify Batch Structure
    batch = next(iter(train_loader))
    print(f"Batch features shape: {batch['features'].shape}")  # (B, C, L)
    print(f"Batch masks shape:    {batch['masks'].shape}")  # (B, L)
    print(f"Batch targets shape:  {batch['targets'].shape}")  # (B, 2, L)

    # 5. Model Initialization
    print("\n[Model Initialization]")
    model = SEResUNet1D(config)
    model.to(config.DEVICE)

    # Dummy Forward Pass
    print("Running dummy forward pass...")
    feats = batch["features"].to(config.DEVICE)
    with torch.no_grad():
        outputs = model(feats)

    print(f"Output heads: {len(outputs)}")
    print(f"Final output shape: {outputs[-1].shape}")

    # Verify output shape matches input length (U-Net structure)
    assert (
        outputs[-1].shape[2] == feats.shape[2]
    ), f"Output length {outputs[-1].shape[2]} does not match input length {feats.shape[2]}"

    # 6. Training Loop
    print("\n[Training Loop]")
    trainer = Trainer(model, config)

    if val_loader:
        trainer.fit(train_loader, val_loader)
    else:
        print("Skipping training loop (no validation data).")

    # 7. Inference & Submission
    print("\n[Inference]")
    if test_loader:
        trainer.generate_submission(test_loader)

        if os.path.exists(config.SUBMISSION_PATH):
            df_sub = pd.read_csv(config.SUBMISSION_PATH)
            print(f"Submission generated successfully.")
            print(f"Shape: {df_sub.shape}")
            print("Head:")
            print(df_sub.head())

            # Basic validity check
            assert "LatitudeDegrees" in df_sub.columns
            assert "LongitudeDegrees" in df_sub.columns
            assert not df_sub.empty
        else:
            raise RuntimeError("Submission file was not created.")
    else:
        print("Skipping inference (no test data).")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
