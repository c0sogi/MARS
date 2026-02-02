import os
import sys
import pandas as pd
import torch
import numpy as np
import logging
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.preprocess import DicomPreprocessor
from library.dataset import get_dataloaders
from library.model import get_model
from library.train import train_one_epoch, evaluate, generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Configuration Overrides for Demo
    # We redirect all paths to ./working to avoid touching input/metadata
    # and to use a small subset of data for speed.
    DEMO_DIR = os.path.join(Config.ROOT_DIR, "working", "demo")
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.MODEL_DIR = os.path.join(DEMO_DIR, "models")
    Config.LOG_DIR = os.path.join(DEMO_DIR, "logs")

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    Config.TRAIN_META = os.path.join(DEMO_DIR, "train_meta_demo.csv")
    Config.VAL_META = os.path.join(DEMO_DIR, "val_meta_demo.csv")
    Config.TEST_META = os.path.join(DEMO_DIR, "test_meta_demo.csv")

    Config.PROCESSED_TRAIN_PKL = os.path.join(DEMO_DIR, "train_processed.parquet")
    Config.PROCESSED_VAL_PKL = os.path.join(DEMO_DIR, "val_processed.parquet")
    Config.PROCESSED_TEST_PKL = os.path.join(DEMO_DIR, "test_processed.parquet")

    Config.MODEL_SAVE_PATH = os.path.join(Config.MODEL_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce compute load
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # 3. Create Data Subsets
    print("\n[Step 1] Creating Data Subsets...")

    # Load original metadata
    orig_train = pd.read_csv(os.path.join(Config.METADATA_DIR, "train_meta.csv"))
    orig_val = pd.read_csv(os.path.join(Config.METADATA_DIR, "val_meta.csv"))
    orig_test = pd.read_csv(os.path.join(Config.METADATA_DIR, "test_meta.csv"))

    # Sample unique images to ensure we don't split objects of the same image
    def sample_subset(df, n_images):
        unique_ids = df["image_id"].unique()
        selected_ids = unique_ids[:n_images]
        return df[df["image_id"].isin(selected_ids)].copy()

    # Create tiny subsets (10 train, 5 val, 5 test)
    demo_train = sample_subset(orig_train, 10)
    demo_val = sample_subset(orig_val, 5)
    demo_test = sample_subset(orig_test, 5)

    # Save to demo location
    demo_train.to_csv(Config.TRAIN_META, index=False)
    demo_val.to_csv(Config.VAL_META, index=False)
    demo_test.to_csv(Config.TEST_META, index=False)

    print(f"  Train subset: {len(demo_train)} rows")
    print(f"  Val subset:   {len(demo_val)} rows")
    print(f"  Test subset:  {len(demo_test)} rows")

    # 4. Preprocessing Demo
    print("\n[Step 2] Running Preprocessor...")
    # Initialize preprocessor
    preprocessor = DicomPreprocessor()

    # Run preprocessing (reads DICOMs, resizes, saves PNGs to cache, saves Parquet)
    # load_cached_data=False forces it to process our new subset files
    preprocessor.run(load_cached_data=False)

    # Validation
    assert os.path.exists(Config.PROCESSED_TRAIN_PKL), "Processed train parquet missing"
    assert os.path.exists(Config.PROCESSED_VAL_PKL), "Processed val parquet missing"
    assert os.path.exists(Config.PROCESSED_TEST_PKL), "Processed test parquet missing"
    print("  Preprocessing validation passed.")

    # 5. Dataset & DataLoader Demo
    print("\n[Step 3] Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Fetch one batch
    images, targets = next(iter(train_loader))

    # Validation
    print(f"  Batch size: {len(images)}")
    assert len(images) == Config.BATCH_SIZE or len(images) == len(
        demo_train["image_id"].unique()
    ), "Batch size mismatch"
    assert isinstance(images, tuple), "Images should be a tuple"
    assert isinstance(targets, tuple), "Targets should be a tuple"
    assert isinstance(images[0], torch.Tensor), "Image content should be tensor"
    assert "boxes" in targets[0], "Target should contain boxes"
    assert "labels" in targets[0], "Target should contain labels"
    print("  DataLoader validation passed.")

    # 6. Model Initialization Demo
    print("\n[Step 4] Initializing Model...")
    model = get_model(num_classes=Config.NUM_CLASSES, img_size=Config.IMG_SIZE)
    model.to(device)

    # Validation: Check Forward Pass (Loss Calculation)
    print("  Testing model forward pass (Training mode)...")
    model.train()
    images_dev = [img.to(device) for img in images]
    targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images_dev, targets_dev)

    print(f"  Loss keys returned: {list(loss_dict.keys())}")
    assert "loss_classifier" in loss_dict, "Missing classifier loss"
    assert "loss_box_reg" in loss_dict, "Missing box regression loss"

    total_loss = sum(loss for loss in loss_dict.values())
    print(f"  Total Loss: {total_loss.item():.4f}")

    # Backward pass check
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print("  Backward pass successful.")

    # 7. Full Training Loop Demo
    print("\n[Step 5] Running Training Loop (1 Epoch)...")
    logger = get_logger(os.path.join(Config.LOG_DIR, "demo_train.log"))

    # Train
    avg_loss = train_one_epoch(model, optimizer, train_loader, device, 0, logger)
    print(f"  Epoch 0 Average Loss: {avg_loss:.4f}")

    # Evaluate
    print("  Running Evaluation...")
    val_loss = evaluate(model, val_loader, device, logger)
    print(f"  Validation Loss: {val_loss:.4f}")

    # 8. Inference Demo
    print("\n[Step 6] Running Inference & Submission Generation...")
    # Save dummy model state to test loading mechanism in generate_submission if needed,
    # but generate_submission takes the model object directly in this library.
    # However, let's verify the submission file generation.

    generate_submission(model, test_loader, device, logger)

    # Validation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission rows: {len(df_sub)}")
    print(f"  Columns: {list(df_sub.columns)}")

    assert len(df_sub) == len(demo_test), "Submission row count mismatch"
    assert "PredictionString" in df_sub.columns, "Missing PredictionString column"

    # Check format of a prediction string
    pred_string = df_sub.iloc[0]["PredictionString"]
    print(f"  Sample PredictionString: {pred_string}")
    # Should be "class score xmin ymin xmax ymax ..." or "14 1 0 0 1 1"
    parts = pred_string.split()
    assert (
        len(parts) % 6 == 0
    ), "PredictionString format invalid (length not multiple of 6)"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
