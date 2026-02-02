import os
import sys
import shutil
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_manager import (
    get_dataloaders,
    get_test_dataloader,
    CassavaDataset,
    get_transforms,
)
from library.model_factory import get_model
from library.training_engine import get_mixup_fn, train_one_epoch, valid_one_epoch
from library.meta_learner import (
    fit_meta_learner,
    predict_meta_learner,
    generate_submission,
)

# Suppress warnings
warnings.filterwarnings("ignore")


def create_debug_metadata(source_path, dest_path, n_samples=50):
    """Creates a small subset of metadata for debugging/demo purposes."""
    df = pd.read_csv(source_path)
    # Ensure we have enough samples for stratification if possible, or just sample
    if len(df) > n_samples:
        df_subset = df.sample(n=n_samples, random_state=Config.SEED).reset_index(
            drop=True
        )
    else:
        df_subset = df.copy()

    df_subset.to_csv(dest_path, index=False)
    print(f"Debug metadata created at {dest_path} with {len(df_subset)} samples.")
    return df_subset


def main():
    print("Starting Cassava Leaf Disease Classification Pipeline Demo...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for a fast demo run
    Config.SEED = 42
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.OUTPUT_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Setup Logging
    log_path = os.path.join(Config.OUTPUT_DIR, "demo.log")
    logger = get_logger("demo", log_path)
    logger.info("Configuration set for demo run.")

    # Seed everything
    seed_everything(Config.SEED)

    # =========================================================================
    # 2. Data Preparation (Subset)
    # =========================================================================
    logger.info("Preparing debug dataset...")

    # Define path for debug metadata
    debug_train_meta_path = os.path.join(Config.OUTPUT_DIR, "train_debug.csv")

    # Create the subset from the provided metadata
    # We use the existing metadata/train.csv as source
    create_debug_metadata(Config.TRAIN_METADATA, debug_train_meta_path, n_samples=60)

    # Point Config to this new file
    Config.TRAIN_METADATA = debug_train_meta_path

    # Also verify test metadata exists, though we won't modify it as it's read-only input
    # We will use the provided test.csv
    assert os.path.exists(Config.TEST_METADATA), "Test metadata not found!"

    # =========================================================================
    # 3. Data Manager Demonstration
    # =========================================================================
    logger.info("Testing Data Manager...")

    # Test 3.1: CassavaDataset instantiation
    df_debug = pd.read_csv(Config.TRAIN_METADATA)
    dataset = CassavaDataset(
        df_debug, transforms=get_transforms("train"), output_label=True
    )

    # Fetch one sample
    img, label = dataset[0]

    # Validation
    assert isinstance(img, torch.Tensor), "Dataset should return a torch.Tensor image"
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Got {img.shape}"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"
    logger.info(f"Dataset sample check passed. Image shape: {img.shape}")

    # Test 3.2: DataLoader generation
    # We use fold_id=0. Since we have ~60 samples and 5 folds, val set should be ~12 samples.
    train_loader, val_loader = get_dataloaders(fold_id=0, load_cached_data=False)

    # Validation
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"

    batch_img, batch_label = next(iter(train_loader))
    assert batch_img.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    logger.info(
        f"DataLoaders created successfully. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    # =========================================================================
    # 4. Model Factory Demonstration
    # =========================================================================
    logger.info("Testing Model Factory...")

    # We pick one architecture from the list to demonstrate
    model_name = "vit_base_patch16_384"
    logger.info(f"Initializing model: {model_name}")

    model = get_model(
        model_name, pretrained=False
    )  # False for speed in demo, usually True
    model.to(Config.DEVICE)

    # Validation: Forward pass with dummy input
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Got {output.shape}"
    logger.info("Model initialized and forward pass verified.")

    # =========================================================================
    # 5. Training Engine Demonstration
    # =========================================================================
    logger.info("Testing Training Engine...")

    # Setup training components
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    loss_fn = nn.CrossEntropyLoss()  # Standard CE for validation
    train_loss_fn = (
        nn.CrossEntropyLoss()
    )  # Using CE for demo simplicity (SoftTargetCE requires mixup targets)
    scaler = GradScaler()
    mixup_fn = get_mixup_fn()

    # 5.1 Train One Epoch
    logger.info("Running training epoch...")
    # Note: In a real run, we use SoftTargetCrossEntropy with Mixup.
    # For this demo, to ensure assertions pass easily with the complex mixup return,
    # we will use the provided functions but be aware mixup changes targets.

    # We need to ensure the loss function matches the target shape.
    # The provided train_one_epoch assumes SoftTargetCrossEntropy if mixup is used.
    # Let's use the timm SoftTargetCrossEntropy as imported in training_engine.
    from timm.loss import SoftTargetCrossEntropy

    train_loss_fn = SoftTargetCrossEntropy()

    avg_train_loss = train_one_epoch(
        epoch=0,
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=Config.DEVICE,
        loss_fn=train_loss_fn,
        scaler=scaler,
        mixup_fn=mixup_fn,
    )

    assert not np.isnan(avg_train_loss), "Training loss is NaN"
    logger.info(f"Training epoch complete. Loss: {avg_train_loss:.4f}")

    # 5.2 Valid One Epoch
    logger.info("Running validation epoch...")
    val_loss, val_acc, val_preds, val_targets = valid_one_epoch(
        epoch=0, model=model, loader=val_loader, device=Config.DEVICE, loss_fn=loss_fn
    )

    # Validation
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_acc <= 1.0, "Validation accuracy out of bounds"
    assert val_preds.shape == (
        len(val_loader.dataset),
        Config.NUM_CLASSES,
    ), "Prediction shape mismatch"
    logger.info(f"Validation epoch complete. Acc: {val_acc:.4f}")

    # Save the demo model weights (simulating a saved checkpoint)
    model_path = os.path.join(Config.OUTPUT_DIR, "demo_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info("Model checkpoint saved.")

    # =========================================================================
    # 6. Meta Learner Demonstration
    # =========================================================================
    logger.info("Testing Meta Learner...")

    # 6.1 Fit Meta Learner
    # Simulate OOF predictions for 2 models to demonstrate stacking
    # We use the val_targets from the previous step as ground truth
    n_samples = len(val_targets)

    # Create synthetic OOF predictions (random probabilities)
    # Model A OOF
    oof_a = np.random.rand(n_samples, Config.NUM_CLASSES)
    oof_a = oof_a / oof_a.sum(axis=1, keepdims=True)

    # Model B OOF
    oof_b = np.random.rand(n_samples, Config.NUM_CLASSES)
    oof_b = oof_b / oof_b.sum(axis=1, keepdims=True)

    oof_list = [oof_a, oof_b]

    meta_model = fit_meta_learner(oof_list, val_targets, save_model=True)

    assert meta_model is not None, "Meta learner training failed"
    logger.info("Meta learner trained successfully.")

    # 6.2 Predict and Generate Submission
    # Simulate Test Predictions
    # Let's load the test metadata to get image IDs
    df_test = pd.read_csv(Config.TEST_METADATA)
    test_ids = df_test["image_id"].values
    n_test = len(df_test)

    # Synthetic test preds for Model A and B
    test_pred_a = np.random.rand(n_test, Config.NUM_CLASSES)
    test_pred_a = test_pred_a / test_pred_a.sum(axis=1, keepdims=True)

    test_pred_b = np.random.rand(n_test, Config.NUM_CLASSES)
    test_pred_b = test_pred_b / test_pred_b.sum(axis=1, keepdims=True)

    test_preds_list = [test_pred_a, test_pred_b]

    # Generate Submission
    generate_submission(test_preds_list, meta_model, test_ids)

    # Validation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert sub_df.shape == (
        n_test,
        2,
    ), f"Submission shape mismatch. Expected ({n_test}, 2), got {sub_df.shape}"
    assert list(sub_df.columns) == ["image_id", "label"], "Submission columns mismatch"

    logger.info("Submission generated and verified.")

    print("\nAll demonstration steps completed successfully!")


if __name__ == "__main__":
    main()
