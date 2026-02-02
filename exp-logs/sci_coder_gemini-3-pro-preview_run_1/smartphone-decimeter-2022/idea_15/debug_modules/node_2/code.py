import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_processing import get_data, load_and_process_dataset
from library.dataset import get_datasets
from library.model import AttentionGatedResUNet1D
from library.loss import MultiScaleMAELoss
from library.trainer import Trainer, generate_submission


def main():
    # 1. Setup and Configuration
    print(">>> Setting up demonstration environment...")
    seed_everything(42)

    # Define temporary paths for mini-metadata to ensure quick execution
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train_meta.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val_meta.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test_meta.csv")

    # Create mini-metadata files by sampling the original metadata
    # We select a single drive to ensure we have enough sequential data for the windowing
    print(">>> Creating mini-metadata files...")

    full_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    # Pick one drive for training
    train_drive = full_train["drive_id"].unique()[0]
    mini_train = full_train[full_train["drive_id"] == train_drive].head(
        500
    )  # 500 samples
    mini_train.to_csv(mini_train_path, index=False)

    full_val = pd.read_csv(Config.VAL_METADATA_PATH)
    if not full_val.empty:
        val_drive = full_val["drive_id"].unique()[0]
        mini_val = full_val[full_val["drive_id"] == val_drive].head(100)
    else:
        # Fallback if val is empty (though generate_metadata ensures it isn't usually)
        mini_val = mini_train.iloc[:50].copy()
    mini_val.to_csv(mini_val_path, index=False)

    full_test = pd.read_csv(Config.TEST_METADATA_PATH)
    # Pick one drive for testing
    test_drive = full_test["drive_id"].unique()[0]
    mini_test = full_test[full_test["drive_id"] == test_drive].head(100)
    mini_test.to_csv(mini_test_path, index=False)

    # Monkey-patch Config to use these mini files and reduce training load
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print(f"    Train samples: {len(mini_train)}")
    print(f"    Val samples:   {len(mini_val)}")
    print(f"    Test samples:  {len(mini_test)}")

    # 2. Data Processing
    print("\n>>> 2. Processing Data...")
    # Force reload to process the mini datasets
    train_df = load_and_process_dataset(
        Config.TRAIN_METADATA_PATH, "mini_train", load_cached_data=False
    )
    val_df = load_and_process_dataset(
        Config.VAL_METADATA_PATH, "mini_val", load_cached_data=False
    )
    test_df = load_and_process_dataset(
        Config.TEST_METADATA_PATH, "mini_test", load_cached_data=False
    )

    # Verification
    assert not train_df.empty, "Processed training dataframe is empty"
    assert "target_east" in train_df.columns, "Target columns missing in train_df"
    assert "Cn0DbHz_mean" in train_df.columns, "Feature columns missing in train_df"
    print("    Data processing successful. Train shape:", train_df.shape)

    # 3. Dataset Creation
    print("\n>>> 3. Creating Datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(train_df, val_df, test_df)

    # Verify Dataset Item
    sample_item = train_dataset[0]
    features = sample_item["features"]
    targets = sample_item["targets"]  # List of multi-scale targets
    mask = sample_item["mask"]

    print(f"    Feature shape: {features.shape} (Channels, Length)")
    print(f"    Target scales: {len(targets)}")
    print(f"    Target[0] shape: {targets[0].shape}")

    assert features.dim() == 2, "Features should be 2D (C, L)"
    assert (
        features.shape[0] == Config.NUM_FEATURES
    ), f"Expected {Config.NUM_FEATURES} features"
    assert len(targets) == 4, "Expected 4 scales for deep supervision"

    # 4. Model Initialization
    print("\n>>> 4. Initializing Model...")
    device = Config.DEVICE
    model = AttentionGatedResUNet1D().to(device)

    # Verify Forward Pass
    # Use batch size > 1 to avoid BatchNorm errors (Cite debug_lesson_16)
    dummy_input = features.unsqueeze(0).repeat(2, 1, 1).to(device)
    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"    Model output list length: {len(outputs)}")
    print(f"    Output[0] shape: {outputs[0].shape}")

    assert len(outputs) == 4, "Model should return 4 outputs for deep supervision"
    assert outputs[0].shape == (
        1,
        Config.NUM_CLASSES,
        features.shape[1],
    ), f"Output shape mismatch. Expected (1, {Config.NUM_CLASSES}, {features.shape[1]}), got {outputs[0].shape}"

    # 5. Loss Calculation
    print("\n>>> 5. Computing Loss...")
    criterion = MultiScaleMAELoss().to(device)

    # Prepare dummy targets and mask on device
    dummy_targets = [t.unsqueeze(0).repeat(2, 1, 1).to(device) for t in targets]
    dummy_mask = mask.unsqueeze(0).repeat(2, 1).to(device)

    loss = criterion(outputs, dummy_targets, dummy_mask)
    print(f"    Calculated Loss: {loss.item()}")
    assert loss.item() >= 0, "Loss should be non-negative"

    # 6. Training Loop
    print("\n>>> 6. Running Training Loop...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=2,
    )

    trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("    Training finished and model saved.")

    # 7. Inference and Submission
    print("\n>>> 7. Generating Submission...")
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Load best model state
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    generate_submission(model, test_loader, device)

    assert os.path.exists(
        Config.SUBMISSION_SAVE_PATH
    ), "Submission file was not created."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_SAVE_PATH)
    print(f"    Submission rows: {len(sub_df)}")
    print(f"    Submission columns: {sub_df.columns.tolist()}")

    expected_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"
    assert not sub_df.empty, "Submission file is empty"

    print("\n>>> Demonstration Complete Successfully!")


if __name__ == "__main__":
    main()
