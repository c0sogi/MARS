import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_score, generate_model_soup
from library.data import (
    get_data_loaders,
    get_test_loader,
    DogCatDataset,
    get_transforms,
)
from library.models import get_model
from library.engine import (
    train_one_epoch,
    validate,
    inference_fn,
    mixup_data,
    cutmix_data,
)


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Override Config for Speed and Demo purposes
    print("\n[1] Configuring environment for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Very small subset
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.EPOCHS = 1

    # Ensure working directories exist (Config creates them on import, but good to double check)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, BATCH_SIZE=8, SUBSET_SIZE=32")

    # 2. Data Loading Demonstration
    print("\n[2] Verifying Data Loading...")
    # We use fold 0. This will load metadata/train.csv and metadata/val.csv
    train_loader, val_loader = get_data_loaders(fold_id=0)

    # Check Train Loader
    try:
        images, labels = next(iter(train_loader))
        print(
            f"Train Batch - Images Shape: {images.shape}, Labels Shape: {labels.shape}"
        )

        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), "Incorrect image tensor shape"
        assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label tensor shape"
        assert images.dtype == torch.float32, "Images should be float32"
        assert labels.dtype == torch.float32, "Labels should be float32 (for BCE)"
        print("Data Loader assertions passed.")
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # 3. Augmentation Logic Demonstration
    print("\n[3] Verifying Augmentation Logic (Mixup/CutMix)...")
    dummy_imgs = torch.randn(4, 3, 224, 224).to(Config.DEVICE)
    dummy_lbls = torch.tensor([0.0, 1.0, 0.0, 1.0]).to(Config.DEVICE)

    # Test Mixup
    mixed_imgs, mixed_lbls = mixup_data(
        dummy_imgs, dummy_lbls, alpha=1.0, device=Config.DEVICE
    )
    assert mixed_imgs.shape == dummy_imgs.shape, "Mixup altered image shape incorrectly"
    assert mixed_lbls.shape == dummy_lbls.shape, "Mixup altered label shape incorrectly"

    # Test CutMix
    cut_imgs, cut_lbls = cutmix_data(
        dummy_imgs, dummy_lbls, alpha=1.0, device=Config.DEVICE
    )
    assert cut_imgs.shape == dummy_imgs.shape, "CutMix altered image shape incorrectly"
    assert cut_lbls.shape == dummy_lbls.shape, "CutMix altered label shape incorrectly"
    print("Augmentation functions executed successfully.")

    # 4. Model Initialization Demonstration
    print("\n[4] Verifying Model Initialization...")
    # We use 'convnext_small.fb_in22k' as defined in Config, but pretrained=False for speed/offline safety
    model_name = Config.MODELS[0]
    print(f"Creating model: {model_name}")
    model = get_model(model_name, pretrained=False)
    model.to(Config.DEVICE)

    # Forward pass check
    with torch.no_grad():
        # Use the batch we fetched earlier
        output = model(images.to(Config.DEVICE))
        # Expected output shape: [Batch_Size, 1] (Binary classification logits)
        print(f"Model Output Shape: {output.shape}")
        assert output.shape == (
            Config.BATCH_SIZE,
            1,
        ), f"Expected output shape {(Config.BATCH_SIZE, 1)}, got {output.shape}"
    print("Model initialization and forward pass successful.")

    # 5. Training Engine Demonstration
    print("\n[5] Verifying Training and Validation Engine...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train one epoch
    print("Running train_one_epoch...")
    train_loss = train_one_epoch(model, optimizer, train_loader, Config.DEVICE, epoch=1)
    print(f"Train Loss returned: {train_loss}")
    assert isinstance(train_loss, float), "train_one_epoch should return a float loss"
    assert train_loss > 0, "Train loss should be positive (usually)"

    # Validate
    print("Running validate...")
    val_loss, val_score = validate(model, val_loader, Config.DEVICE)
    print(f"Val Loss: {val_loss}, Val LogLoss: {val_score}")
    assert isinstance(val_loss, float), "Validation loss should be float"
    assert isinstance(val_score, float), "Validation score should be float"
    print("Engine functions executed successfully.")

    # 6. Inference Demonstration
    print("\n[6] Verifying Inference Function...")
    # inference_fn expects (images, ids)
    # val_loader returns (images, labels). For demo, we can just use val_loader and ignore that the second item is label,
    # treating it as an ID for the dictionary key.
    # However, to be strict, let's use the test loader logic or just mock it.
    # Let's use the actual test loader structure since get_test_loader is provided.

    test_loader = get_test_loader()
    # Fetch one batch to ensure it works
    test_imgs, test_ids = next(iter(test_loader))
    print(f"Test Batch - Images: {test_imgs.shape}, IDs: {test_ids.shape}")

    # Run inference
    # We limit the loader to just one batch for speed by mocking the loader iterator if needed,
    # but since DEBUG_SUBSET_SIZE is 32 and Batch is 8, it's only 4 batches. Fast enough.
    preds = inference_fn(model, test_loader, Config.DEVICE)

    print(f"Inference predictions count: {len(preds)}")
    assert len(preds) > 0, "Inference returned empty predictions"
    # Check first key
    first_id = list(preds.keys())[0]
    assert isinstance(
        preds[first_id], (float, np.float32, np.float64)
    ), "Prediction values should be floats"
    print("Inference function executed successfully.")

    # 7. Model Soup Demonstration
    print("\n[7] Verifying Model Soup Generation...")
    # Save current model as two different checkpoints
    ckpt_path_1 = os.path.join(Config.CHECKPOINT_DIR, "demo_ckpt_1.pth")
    ckpt_path_2 = os.path.join(Config.CHECKPOINT_DIR, "demo_ckpt_2.pth")
    soup_path = os.path.join(Config.CHECKPOINT_DIR, "demo_soup.pth")

    # Modify weights slightly for the second checkpoint to simulate difference
    torch.save(model.state_dict(), ckpt_path_1)

    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.01)
    torch.save(model.state_dict(), ckpt_path_2)

    # Generate Soup
    generate_model_soup([ckpt_path_1, ckpt_path_2], soup_path)

    # Verify file exists and loads
    assert os.path.exists(soup_path), "Soup file was not created"
    soup_state = torch.load(soup_path)
    assert isinstance(soup_state, dict), "Loaded soup state is not a dict"
    print("Model Soup generation successful.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
