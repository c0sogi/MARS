import os
import torch
import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Import from provided library files
from library.config import Config
from library.dataset import (
    BSONDataset,
    get_transforms,
    train_collate_fn,
    eval_collate_fn,
)
from library.model import get_model
from library.engine import (
    CategoryMapper,
    train_one_epoch,
    validate,
    predict,
    seed_everything,
)


def main():
    print("==== Starting Demonstration Script ====")

    # 1. Setup and Configuration Override
    # We override defaults to run a fast check on a small subset of data
    Config.setup()
    seed_everything(Config.SEED)

    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 200  # Small subset for rapid execution
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 2
    Config.NUM_EPOCHS = 1

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Subset Size: {Config.DEBUG_SUBSET_SIZE}")

    # 2. Verify Category Mapper
    print("\n[1/5] Initializing Category Mapper...")
    mapper = CategoryMapper()
    print(f"Total Classes: {mapper.num_classes}")

    # Assert we have the expected number of classes
    if mapper.num_classes != 5270:
        raise AssertionError(f"Expected 5270 classes, found {mapper.num_classes}")

    # Test mapping logic
    test_cat_id = mapper.categories[0]
    test_idx = mapper.to_idx([test_cat_id])[0].item()
    assert test_idx == 0, "Mapping logic error: First category should map to index 0"
    assert mapper.to_cat([0])[0] == test_cat_id, "Reverse mapping logic error"

    # 3. Verify Dataset and DataLoader
    print("\n[2/5] Initializing Datasets & Loaders...")

    # Train Dataset
    train_dataset = BSONDataset(
        Config.TRAIN_META,
        mode="train",
        transform=get_transforms("train", Config.IMG_SIZE),
        debug=Config.DEBUG,
    )

    # Val Dataset
    val_dataset = BSONDataset(
        Config.VAL_META,
        mode="val",
        transform=get_transforms("val", Config.IMG_SIZE),
        debug=Config.DEBUG,
    )

    print(f"Train Dataset Length (Debug): {len(train_dataset)}")
    print(f"Val Dataset Length (Debug): {len(val_dataset)}")

    # Verify Train Item Structure
    img, label = train_dataset[0]
    if img.shape != (3, Config.IMG_SIZE, Config.IMG_SIZE):
        raise AssertionError(f"Train image shape mismatch. Got {img.shape}")
    if not isinstance(label, int):
        raise AssertionError(f"Train label should be int. Got {type(label)}")

    # Verify Val Item Structure
    imgs_tensor, label, pid = val_dataset[0]
    # Val returns (N_imgs, C, H, W)
    if imgs_tensor.ndim != 4 or imgs_tensor.shape[1:] != (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ):
        raise AssertionError(
            f"Val images tensor shape mismatch. Got {imgs_tensor.shape}"
        )

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=train_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=eval_collate_fn,
    )

    # Verify Batch Structure
    batch_imgs, batch_labels = next(iter(train_loader))
    if batch_imgs.shape != (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE):
        raise AssertionError(f"Train batch image shape mismatch: {batch_imgs.shape}")
    if batch_labels.shape != (Config.BATCH_SIZE,):
        raise AssertionError(f"Train batch label shape mismatch: {batch_labels.shape}")

    # 4. Verify Model Initialization
    print("\n[3/5] Initializing Model...")
    model = get_model(num_classes=mapper.num_classes, device=device)

    # Basic check of model output shape
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    if output.shape != (2, mapper.num_classes):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, {mapper.num_classes}), got {output.shape}"
        )
    print("Model initialized and forward pass verified.")

    # 5. Verify Training and Validation Loop
    print("\n[4/5] Running Training & Validation Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler()
    loss_fn = torch.nn.CrossEntropyLoss()

    # Run Train
    start_time = time.time()
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, None, scaler, loss_fn, device, mapper
    )
    print(f"Train Epoch Done. Time: {time.time() - start_time:.2f}s")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Run Val
    start_time = time.time()
    val_loss, val_acc = validate(model, val_loader, loss_fn, device, mapper)
    print(f"Val Epoch Done. Time: {time.time() - start_time:.2f}s")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 6. Verify Prediction / Inference
    print("\n[5/5] Running Inference on Test Subset...")

    test_dataset = BSONDataset(
        Config.TEST_META,
        mode="test",
        transform=get_transforms("test", Config.IMG_SIZE),
        debug=Config.DEBUG,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=eval_collate_fn,
    )

    df_submission = predict(model, test_loader, device, mapper)

    print(f"Prediction complete. Generated {len(df_submission)} predictions.")
    print("Sample predictions:")
    print(df_submission.head())

    # Verify Submission Format
    required_cols = ["_id", "category_id"]
    if not all(col in df_submission.columns for col in required_cols):
        raise AssertionError(f"Submission missing columns. Expected {required_cols}")

    if len(df_submission) != len(test_dataset):
        raise AssertionError(
            f"Prediction count mismatch. Expected {len(test_dataset)}, got {len(df_submission)}"
        )

    # Save
    output_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    df_submission.to_csv(output_path, index=False)
    print(f"\nDemo submission saved to: {output_path}")
    print("==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
