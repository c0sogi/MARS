import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil

# Import provided library components
from library.config import Config
from library.utils import set_seed, get_hierarchy_mappings
from library.dataset import PlantDataset, get_dataloader
from library.model import HierarchicalEfficientNet
from library.train import train_one_epoch, validate
from library.predict import inference


def main():
    print("Starting demonstration of Plant Classification Library...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demonstration
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Enable Debug mode to use small data subsets (e.g., 32 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 32

    # Set a temporary working directory for this run
    Config.WORKING_DIR = "./working/demo_script_output"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths to point to the new working directory
    Config.HIERARCHY_MAPPING_PATH = os.path.join(
        Config.WORKING_DIR, "hierarchy_mappings.parquet"
    )
    Config.CHECKPOINT_STAGE_1 = os.path.join(
        Config.WORKING_DIR, "stage_1_checkpoint.pth"
    )
    Config.CHECKPOINT_STAGE_2 = os.path.join(
        Config.WORKING_DIR, "stage_2_checkpoint.pth"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Reduce training parameters
    Config.STAGE_1_EPOCHS = 1
    Config.STAGE_1_BATCH_SIZE = 8
    Config.STAGE_2_BATCH_SIZE = 8
    Config.NUM_WORKERS = 2  # Reduce workers for small data

    set_seed(Config.SEED)
    print("Configuration updated. Debug mode enabled.")

    # -------------------------------------------------------------------------
    # 2. Hierarchy Mapping Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Hierarchy Mappings...")

    # This parses the JSON or loads from cache to map Species -> Genus -> Family
    s2g, s2f, num_genera, num_families = get_hierarchy_mappings(load_cached_data=False)

    print(f"   Num Species: {Config.NUM_CLASSES_SPECIES}")
    print(f"   Num Genera:  {num_genera}")
    print(f"   Num Families: {num_families}")

    # Assertions
    assert len(s2g) > 0, "Species to Genus mapping is empty."
    assert len(s2f) > 0, "Species to Family mapping is empty."
    assert num_genera > 0, "Number of genera must be positive."
    assert num_families > 0, "Number of families must be positive."
    print("Hierarchy mapping verification passed.")

    # -------------------------------------------------------------------------
    # 3. Dataset and DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Dataset and DataLoader...")

    # Load metadata dataframe
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Instantiate Dataset
    dataset = PlantDataset(df=df_train, mode="train", transform=None)

    # Verify dataset length (should be capped by DEBUG_SAMPLE_SIZE)
    print(f"   Dataset length (Debug): {len(dataset)}")
    assert (
        len(dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Dataset length mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(dataset)}"

    # Verify single item structure
    img_tensor, targets = dataset[0]
    print(f"   Sample 0 Image Shape: {img_tensor.shape}")
    print(f"   Sample 0 Targets: {targets}")

    assert img_tensor.shape[0] == 3, "Image tensor should have 3 channels."
    assert "species" in targets, "Targets missing 'species' key."
    assert "genus" in targets, "Targets missing 'genus' key."
    assert "family" in targets, "Targets missing 'family' key."

    # Instantiate DataLoader
    loader = get_dataloader(
        df_train,
        mode="train",
        batch_size=Config.STAGE_1_BATCH_SIZE,
        image_size=Config.STAGE_1_RES,
    )

    # Fetch one batch
    images, batch_targets = next(iter(loader))
    print(f"   Batch Image Shape: {images.shape}")
    assert images.shape == (
        Config.STAGE_1_BATCH_SIZE,
        3,
        Config.STAGE_1_RES,
        Config.STAGE_1_RES,
    ), "Incorrect batch image shape."
    assert (
        batch_targets["species"].shape[0] == Config.STAGE_1_BATCH_SIZE
    ), "Incorrect batch target shape."
    print("Dataset and DataLoader verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = Config.DEVICE
    model = HierarchicalEfficientNet(pretrained=False)  # False for speed
    model.to(device)

    # Forward pass with the batch loaded previously
    images = images.to(device)
    outputs = model(images)

    print(f"   Model Output Keys: {outputs.keys()}")

    # Verify output shapes
    assert outputs["species"].shape == (
        Config.STAGE_1_BATCH_SIZE,
        Config.NUM_CLASSES_SPECIES,
    )
    assert outputs["genus"].shape == (Config.STAGE_1_BATCH_SIZE, num_genera)
    assert outputs["family"].shape == (Config.STAGE_1_BATCH_SIZE, num_families)
    print("Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Training Step...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = None  # Skip scheduler for simple demo

    # Run one epoch of training
    train_loss = train_one_epoch(
        model, loader, optimizer, scheduler, device, criterion, epoch=0
    )
    print(f"   Training Loss (1 Epoch): {train_loss:.4f}")
    assert isinstance(train_loss, float), "Train loss should be a float."

    # Run validation
    val_loader = get_dataloader(
        pd.read_csv(Config.VAL_METADATA_PATH),
        mode="valid",
        batch_size=Config.STAGE_1_BATCH_SIZE,
        image_size=Config.STAGE_1_RES,
    )
    val_loss, val_f1 = validate(model, val_loader, device, criterion)
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation F1: {val_f1:.4f}")

    assert isinstance(val_loss, float), "Validation loss should be a float."
    assert 0.0 <= val_f1 <= 1.0, "F1 score must be between 0 and 1."
    print("Training loop demonstration passed.")

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Demonstrating Inference...")

    # Save the current model state as a checkpoint to simulate a trained model
    torch.save(model.state_dict(), Config.CHECKPOINT_STAGE_2)
    print(f"   Saved dummy checkpoint to {Config.CHECKPOINT_STAGE_2}")

    # Run inference
    # Note: Config.DEBUG=True limits the test set size as well
    inference(
        checkpoint_path=Config.CHECKPOINT_STAGE_2, batch_size=Config.STAGE_2_BATCH_SIZE
    )

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"   Submission file created with {len(sub_df)} rows.")
        print(f"   First 5 rows:\n{sub_df.head()}")

        # Check format
        assert list(sub_df.columns) == [
            "Id",
            "Predicted",
        ], "Submission columns mismatch."
        assert (
            len(sub_df) == Config.DEBUG_SAMPLE_SIZE
        ), f"Submission length mismatch. Expected {Config.DEBUG_SAMPLE_SIZE} (Debug), got {len(sub_df)}"
        print("Inference demonstration passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
