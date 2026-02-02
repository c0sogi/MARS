import os
import sys
import pandas as pd
import torch
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_dataloaders
from library.model import get_model, get_optimizer, get_scheduler, get_loss_fn
from library.train import train_one_epoch, validate
from library.inference import run_inference


def create_mini_datasets():
    """
    Creates small subsets of the original metadata to ensure the demo runs fast.
    """
    print("Creating mini datasets for demonstration...")

    # Define paths for mini datasets
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    # Load original metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Sample a small subset (e.g., 16 samples to fit 2 batches of size 8)
    # We ensure we have both classes in the train/val subset
    train_subset = (
        train_df.groupby("label").apply(lambda x: x.head(8)).reset_index(drop=True)
    )
    val_subset = (
        val_df.groupby("label").apply(lambda x: x.head(8)).reset_index(drop=True)
    )
    test_subset = test_df.head(16)

    # Save mini datasets
    train_subset.to_csv(mini_train_path, index=False)
    val_subset.to_csv(mini_val_path, index=False)
    test_subset.to_csv(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("demo_script")
    logger.info("Starting library usage demonstration...")

    # 2. Prepare Data & Override Config
    # We override Config attributes to use the mini datasets and speed up execution
    mini_train, mini_val, mini_test = create_mini_datasets()

    Config.TRAIN_CSV = mini_train
    Config.VAL_CSV = mini_val
    Config.TEST_CSV = mini_test
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.PRETRAINED = False  # Skip downloading weights for speed

    # 3. Demonstrate Data Loading
    logger.info("--- Testing Data Loading ---")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv=Config.TRAIN_CSV,
        val_csv=Config.VAL_CSV,
        test_csv=Config.TEST_CSV,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify Train Loader
    images, labels = next(iter(train_loader))
    logger.info(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"
    assert labels.dtype == torch.long, "Labels must be LongTensor"

    # Verify Test Loader (returns images and IDs)
    test_images, test_ids = next(iter(test_loader))
    logger.info(f"Test Batch - Images: {test_images.shape}, IDs: {len(test_ids)}")
    assert len(test_ids) == Config.BATCH_SIZE, "Incorrect number of test IDs"

    # 4. Demonstrate Model Initialization
    logger.info("--- Testing Model Initialization ---")
    # Using pretrained=False for speed/offline capability in demo
    model = get_model(pretrained=False, device=Config.DEVICE)

    # Verify Model Output Shape
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    logger.info(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch"

    # 5. Demonstrate Training Components
    logger.info("--- Testing Training Components ---")
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    criterion = get_loss_fn()

    # Run one epoch of training
    logger.info("Running train_one_epoch...")
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, Config.DEVICE
    )
    logger.info(f"Train Loss: {train_loss:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"
    assert train_loss > 0, "Training loss should be positive"

    # Run validation
    logger.info("Running validation...")
    val_loss = validate(model, val_loader, Config.DEVICE)
    logger.info(f"Validation Log Loss: {val_loss:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"

    # 6. Demonstrate Inference Pipeline
    logger.info("--- Testing Inference Pipeline ---")

    # Save the current model state as "best_model.pth"
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved dummy model checkpoint to {model_path}")

    # Run full inference function
    submission_path = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    run_inference(
        checkpoint_path=model_path,
        output_path=submission_path,
        device=Config.DEVICE,
        use_tta=False,  # Disable TTA for speed in demo
        batch_size=Config.BATCH_SIZE,
    )

    # Verify Submission
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    logger.info(f"Submission generated with {len(sub_df)} rows.")

    # Check submission content
    assert list(sub_df.columns) == ["id", "label"], "Submission columns mismatch"
    assert len(sub_df) == len(pd.read_csv(mini_test)), "Submission row count mismatch"
    assert (
        sub_df["label"].min() >= 0 and sub_df["label"].max() <= 1
    ), "Probabilities out of range"

    logger.info("Demonstration completed successfully.")


if __name__ == "__main__":
    main()
