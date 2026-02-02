import os
import pandas as pd
import torch
import numpy as np
import shutil
import sys

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_class_weights
from library.data import get_bag_loaders, get_test_loader, AppleDataset, get_transforms
from library.model import AppleResNet34, run_training_pipeline, generate_submission_file
from library.engine import train_one_epoch, validate


def main():
    print("==== Starting Demonstration Script ====")

    # 1. Setup Environment and Configuration for Speed
    # We will override the default Config to run a quick demo
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths and parameters
    Config.WORKING_DIR = demo_dir
    Config.OUTPUT_DIR = os.path.join(demo_dir, "output")
    Config.MODELS_DIR = os.path.join(demo_dir, "models")
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    Config.setup()

    # Set hyperparameters for rapid execution
    Config.EPOCHS = 1
    Config.NUM_BAGS = 2  # Train 2 bags to demonstrate the loop, but on small data
    Config.BATCH_SIZE = 4
    Config.DEBUG = True

    seed_everything(Config.SEED)

    # 2. Create a Small Data Subset
    # Instead of using the full metadata, we create small CSVs to ensure the demo finishes quickly.
    print("\n[1] Creating small data subsets for demonstration...")

    full_train = pd.read_csv("./metadata/train_metadata.csv")
    full_val = pd.read_csv("./metadata/val_metadata.csv")
    full_test = pd.read_csv("./metadata/test_metadata.csv")

    # Sample 20 rows for training, 10 for validation, 10 for testing
    # Ensure we have stratified samples if possible, but for demo random is fine
    # We filter to ensure files exist (though metadata should be correct)
    subset_train = full_train.sample(n=32, random_state=42).reset_index(drop=True)
    subset_val = full_val.sample(n=16, random_state=42).reset_index(drop=True)
    subset_test = full_test.sample(n=16, random_state=42).reset_index(drop=True)

    subset_train_path = os.path.join(demo_dir, "train_subset.csv")
    subset_val_path = os.path.join(demo_dir, "val_subset.csv")
    subset_test_path = os.path.join(demo_dir, "test_subset.csv")

    subset_train.to_csv(subset_train_path, index=False)
    subset_val.to_csv(subset_val_path, index=False)
    subset_test.to_csv(subset_test_path, index=False)

    # Update Config to point to these new files
    Config.TRAIN_METADATA_PATH = subset_train_path
    Config.VAL_METADATA_PATH = subset_val_path
    Config.TEST_METADATA_PATH = subset_test_path

    print(f"    Train subset: {len(subset_train)} rows")
    print(f"    Val subset:   {len(subset_val)} rows")
    print(f"    Test subset:  {len(subset_test)} rows")

    # 3. Verify Data Loading Components
    print("\n[2] Verifying Data Loading...")

    # Test Transforms
    transforms = get_transforms("train")
    assert transforms is not None, "Transforms should not be None"

    # Test Dataset directly
    dataset = AppleDataset(subset_train, transforms=transforms)
    img, label = dataset[0]
    print(f"    Sample Image Shape: {img.shape}")
    print(f"    Sample Label: {label}")

    assert img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image dimensions"
    assert label.shape == (4,), "Incorrect label shape (should be 4 classes)"

    # Test DataLoaders (Bag 0)
    # Note: get_bag_loaders combines train and val metadata, then splits.
    train_loader, val_loader = get_bag_loaders(0)

    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"    Batch Image Shape: {batch_imgs.shape}")
    print(f"    Batch Label Shape: {batch_labels.shape}")

    assert batch_imgs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert batch_imgs.shape[1] == 3, "Channel count mismatch"

    # 4. Verify Model Architecture
    print("\n[3] Verifying Model Architecture...")
    device = Config.DEVICE
    model = AppleResNet34(
        num_classes=4, pretrained=False
    )  # No need to download weights for demo
    model.to(device)

    # Forward pass check
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")
    assert output.shape == (2, 4), "Model output shape mismatch"

    # 5. Verify Training Engine (Single Step)
    print("\n[4] Verifying Training Engine...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Run one epoch using the engine function
    loss, auc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"    Single Epoch - Loss: {loss:.4f}, AUC: {auc:.4f}")

    assert not np.isnan(loss), "Loss is NaN"
    assert 0 <= auc <= 1, "AUC out of range"

    # Run validation
    val_loss, val_auc = validate(model, val_loader, criterion, device)
    print(f"    Validation - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # 6. Verify Full Pipeline Execution
    # This runs the loop over bags defined in Config.NUM_BAGS (set to 2)
    print("\n[5] Running Full Training Pipeline (Mini-Mode)...")

    # We need to clear the cache for bags because we changed the dataset
    # The library caches bag indices in Config.CACHE_DIR
    bags_cache = os.path.join(Config.CACHE_DIR, "bags.parquet")
    if os.path.exists(bags_cache):
        os.remove(bags_cache)

    run_training_pipeline()

    # Check if models were saved
    expected_model_0 = os.path.join(Config.MODELS_DIR, "bag_0_best.pth")
    expected_model_1 = os.path.join(Config.MODELS_DIR, "bag_1_best.pth")

    assert os.path.exists(expected_model_0), "Model for bag 0 not found"
    assert os.path.exists(expected_model_1), "Model for bag 1 not found"
    print("    Pipeline completed successfully. Models saved.")

    # 7. Verify Submission Generation
    print("\n[6] Generating Submission...")
    generate_submission_file()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission Shape: {sub_df.shape}")
    print(f"    Submission Columns: {sub_df.columns.tolist()}")

    assert len(sub_df) == len(
        subset_test
    ), f"Submission row count mismatch. Expected {len(subset_test)}, got {len(sub_df)}"
    assert "image_id" in sub_df.columns
    assert "healthy" in sub_df.columns

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
