import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import DogClassifier
from library.engine import train_head_only, train_one_epoch, validate
from library.soup import create_model_soup

if __name__ == "__main__":
    # ==========================================
    # 1. Setup and Configuration Overrides
    # ==========================================
    print("Initializing demonstration...")

    # Override Config for speed
    Config.debug = True
    Config.debug_sample_size = 20  # Very small subset for speed
    Config.batch_size = 4
    Config.num_workers = 2
    Config.warmup_epochs = 1
    Config.finetune_epochs = 1
    Config.n_folds = 2  # Only need to demo logic

    # Ensure working directory exists (handled by Config import, but good practice)
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set seeds
    seed_everything(Config.seed)

    # Initialize Logger
    logger = get_logger("demo", os.path.join(Config.working_dir, "demo.log"))
    logger.info("Configuration configured for rapid demonstration.")

    # ==========================================
    # 2. Dataset and Dataloader Verification
    # ==========================================
    logger.info("--- Testing Data Loading ---")

    # Get dataloaders for Fold 0
    train_loader, val_loader, classes = get_dataloaders(fold=0, load_cached_data=False)

    # Assertions for Dataloaders
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"
    assert (
        len(classes) == Config.num_classes
    ), f"Expected {Config.num_classes} classes, got {len(classes)}"

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))

    logger.info(f"Batch shapes - Images: {images.shape}, Labels: {labels.shape}")

    # Verify Image Shape: [Batch_Size, Channels, Height, Width]
    assert images.shape == (
        Config.batch_size,
        3,
        Config.image_size,
        Config.image_size,
    ), f"Incorrect image shape: {images.shape}"

    # Verify Label Shape: [Batch_Size]
    assert labels.shape == (
        Config.batch_size,
    ), f"Incorrect label shape: {labels.shape}"

    logger.info("Data loading verification passed.")

    # ==========================================
    # 3. Model Initialization and Logic Checks
    # ==========================================
    logger.info("--- Testing Model Logic ---")

    device = Config.device
    model = DogClassifier(num_classes=Config.num_classes, pretrained=True)
    model.to(device)

    # Check 1: Freeze Backbone Logic
    model.freeze_backbone()

    # Verify: Head should be trainable, Backbone should be frozen
    # We inspect the classifier (head) and the first layer of the backbone
    classifier_params = list(model.model.get_classifier().parameters())
    backbone_param = list(model.model.parameters())[0]  # Likely stem or first conv

    # Ensure classifier params are trainable
    assert all(
        p.requires_grad for p in classifier_params
    ), "Classifier parameters should be trainable after freeze_backbone"

    # Ensure backbone param is NOT trainable (unless it happens to be part of classifier, which is unlikely for index 0)
    # Note: In some models param 0 might be special, but generally for ConvNeXt it is the stem.
    # To be safe, we check if the param object is NOT in classifier_params
    if id(backbone_param) not in map(id, classifier_params):
        assert not backbone_param.requires_grad, "Backbone parameter should be frozen"

    logger.info("Model backbone freezing logic verified.")

    # Check 2: Unfreeze All Logic
    model.unfreeze_all()
    assert all(
        p.requires_grad for p in model.model.parameters()
    ), "All parameters should be trainable after unfreeze_all"

    logger.info("Model unfreezing logic verified.")

    # Check 3: Forward Pass
    dummy_input = torch.randn(
        Config.batch_size, 3, Config.image_size, Config.image_size
    ).to(device)
    output = model(dummy_input)
    assert output.shape == (
        Config.batch_size,
        Config.num_classes,
    ), f"Output shape mismatch: {output.shape}"

    logger.info("Model forward pass verified.")

    # ==========================================
    # 4. Training Engine Verification
    # ==========================================
    logger.info("--- Testing Training Engine ---")

    # Setup Optimizer
    # Using simple settings for demo
    optimizer_params = model.get_optimizer_params(lr_backbone=1e-5, lr_head=1e-3)
    optimizer = optim.AdamW(optimizer_params, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: Head Warmup
    logger.info("Running Head Warmup (1 epoch)...")
    loss_warmup = train_head_only(model, train_loader, optimizer, device, epoch=1)
    assert isinstance(loss_warmup, float), "train_head_only did not return a float loss"
    assert not np.isnan(loss_warmup), "Warmup loss is NaN"

    # Phase 2: Fine-tuning
    logger.info("Running Fine-tuning (1 epoch)...")
    loss_finetune = train_one_epoch(model, train_loader, optimizer, device, epoch=2)
    assert isinstance(
        loss_finetune, float
    ), "train_one_epoch did not return a float loss"

    # Validation
    logger.info("Running Validation...")
    val_loss, val_preds = validate(model, val_loader, criterion, device)

    assert isinstance(val_loss, float), "Validation loss is not a float"
    assert val_preds.shape == (
        len(val_loader.dataset),
        Config.num_classes,
    ), "Validation predictions shape mismatch"

    logger.info(f"Training cycle verified. Val Loss: {val_loss:.4f}")

    # ==========================================
    # 5. Model Soup Verification
    # ==========================================
    logger.info("--- Testing Model Soup ---")

    # Save the current model as "checkpoint 1"
    ckpt_1_path = os.path.join(Config.working_dir, "ckpt_1.pth")
    torch.save(model.state_dict(), ckpt_1_path)

    # Save it again as "checkpoint 2" (simulating another epoch/fold)
    ckpt_2_path = os.path.join(Config.working_dir, "ckpt_2.pth")
    torch.save(model.state_dict(), ckpt_2_path)

    soup_path = os.path.join(Config.working_dir, "soup_demo.pth")

    # Create Soup
    create_model_soup([ckpt_1_path, ckpt_2_path], soup_path)

    # Verify Soup Exists
    assert os.path.exists(soup_path), "Soup file was not created"

    # Verify Loading Soup
    soup_state = torch.load(soup_path, map_location="cpu")
    model.load_state_dict(soup_state)
    logger.info("Model soup created and loaded successfully.")

    # ==========================================
    # 6. Test Inference Verification
    # ==========================================
    logger.info("--- Testing Inference ---")

    test_loader, test_df = get_test_dataloader(load_cached_data=True)

    # Just run one batch
    model.eval()
    with torch.no_grad():
        test_imgs, test_ids = next(iter(test_loader))
        test_imgs = test_imgs.to(device)
        logits = model(test_imgs)
        probs = torch.softmax(logits, dim=1)

    assert probs.shape == (
        test_imgs.size(0),
        Config.num_classes,
    ), "Test probability shape mismatch"
    assert len(test_ids) == test_imgs.size(0), "Test IDs count mismatch"

    logger.info("Inference verified.")

    print("\nAll demonstrations completed successfully.")
