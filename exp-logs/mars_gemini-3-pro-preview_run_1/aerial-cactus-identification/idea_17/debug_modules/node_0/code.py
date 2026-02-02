import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_datasets, mixup_data
from library.models import CactusModel
from library.engine import train_one_epoch, validate, SWAHandler


def demonstrate_dataset_and_transforms():
    print("\n=== Demonstrating Dataset and Transforms ===")

    # 1. Load Datasets
    # This triggers load_and_cache_data internally
    train_ds, val_ds, test_ds = get_datasets(load_cached_data=True)

    print(f"Train Dataset Length: {len(train_ds)}")
    print(f"Val Dataset Length: {len(val_ds)}")
    print(f"Test Dataset Length: {len(test_ds)}")

    # Assertions
    assert len(train_ds) > 0, "Training dataset should not be empty"
    assert len(val_ds) > 0, "Validation dataset should not be empty"

    # 2. Check Single Item
    img, label, film_input, mtl_target, img_id = train_ds[0]

    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Label: {label}")
    print(f"Sample FiLM Input (Normalized File Size): {film_input}")
    print(f"Sample MTL Target (Log File Size): {mtl_target}")

    # Assertions
    assert img.shape == (
        3,
        32,
        32,
    ), f"Expected image shape (3, 32, 32), got {img.shape}"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"
    assert isinstance(film_input, torch.Tensor), "FiLM input should be a tensor"

    # 3. Demonstrate Mixup
    print("Testing Mixup logic...")
    batch_size = 4
    dummy_imgs = torch.randn(batch_size, 3, 32, 32).to(Config.DEVICE)
    # Create combined targets (Label + MTL) for mixup
    dummy_targets = torch.randn(batch_size, 2).to(Config.DEVICE)

    mixed_imgs, y_a, y_b, lam = mixup_data(
        dummy_imgs, dummy_targets, alpha=0.2, device=Config.DEVICE
    )

    assert mixed_imgs.shape == dummy_imgs.shape, "Mixed images shape mismatch"
    assert y_a.shape == dummy_targets.shape, "Target A shape mismatch"
    assert y_b.shape == dummy_targets.shape, "Target B shape mismatch"
    print("Mixup test passed.")

    return train_ds, val_ds


def demonstrate_models():
    print("\n=== Demonstrating Model Architectures ===")

    backbones = ["RepVGG", "ResNet", "NeXt"]
    batch_size = 2
    dummy_img = torch.randn(batch_size, 3, 32, 32).to(Config.DEVICE)
    dummy_film = torch.randn(batch_size, 1).to(Config.DEVICE)

    for backbone in backbones:
        print(f"Testing backbone: {backbone}")
        model = CactusModel(backbone_name=backbone, num_classes=1).to(Config.DEVICE)
        model.eval()

        # Forward pass
        with torch.no_grad():
            logits, quality_pred = model(dummy_img, dummy_film)

        print(f"  Logits shape: {logits.shape}")
        print(f"  Aux shape: {quality_pred.shape}")

        # Assertions
        assert logits.shape == (batch_size, 1), f"Logits shape mismatch for {backbone}"
        assert quality_pred.shape == (
            batch_size,
            1,
        ), f"Aux output shape mismatch for {backbone}"

        # RepVGG Deploy Test
        if backbone == "RepVGG":
            print("  Testing RepVGG switch_to_deploy...")
            model.switch_to_deploy()
            with torch.no_grad():
                logits_deploy, _ = model(dummy_img, dummy_film)
            # Outputs might differ slightly due to float precision, but shapes must match
            assert logits_deploy.shape == logits.shape
            print("  RepVGG deploy switch successful.")


def demonstrate_training_engine(train_ds, val_ds):
    print("\n=== Demonstrating Training Engine ===")

    # Create small subsets for speed
    subset_indices = list(range(32))
    train_subset = Subset(train_ds, subset_indices)
    val_subset = Subset(val_ds, subset_indices)

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=0)

    # Initialize Model, Optimizer, Loss
    model = CactusModel(backbone_name="ResNet", num_classes=1).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_aux = nn.MSELoss()

    # Train One Epoch
    print("Running train_one_epoch...")
    loss, cls_loss, aux_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion_cls=criterion_cls,
        criterion_aux=criterion_aux,
        device=Config.DEVICE,
        epoch=0,
        mixup_alpha=0.2,
        mtl_weight=0.1,
    )

    print(
        f"Training Results - Total Loss: {loss:.4f}, Cls Loss: {cls_loss:.4f}, Aux Loss: {aux_loss:.4f}"
    )
    assert not np.isnan(loss), "Training loss is NaN"

    # Validate
    print("Running validate...")
    val_loss, val_auc = validate(
        model=model,
        val_loader=val_loader,
        criterion_cls=criterion_cls,
        criterion_aux=criterion_aux,
        device=Config.DEVICE,
        mtl_weight=0.1,
    )

    print(f"Validation Results - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_auc <= 1.0, "AUC score out of range"

    return model, train_loader, optimizer


def demonstrate_swa(model, train_loader, optimizer):
    print("\n=== Demonstrating SWA Handler ===")

    # Temporarily override config for demonstration
    Config.USE_SWA = True
    Config.SWA_START_EPOCH = 0  # Start immediately
    Config.SWA_LR = 1e-4

    swa_handler = SWAHandler(model, optimizer, Config)

    # Simulate an epoch end
    print("Simulating epoch end (SWA update)...")
    swa_handler.on_epoch_end(model, epoch=0)

    # Check if SWA model exists
    swa_model = swa_handler.get_model()
    assert swa_model is not None, "SWA model should be initialized"

    # Finalize (BN Update)
    print("Finalizing SWA (BN Update)...")
    # We use the small train_loader from previous step
    swa_handler.finalize(train_loader)

    # Verify SWA model inference
    swa_model.eval()
    dummy_img = torch.randn(2, 3, 32, 32).to(Config.DEVICE)
    dummy_film = torch.randn(2, 1).to(Config.DEVICE)

    with torch.no_grad():
        logits, _ = swa_model(dummy_img, dummy_film)

    assert logits.shape == (2, 1), "SWA model inference shape mismatch"
    print("SWA demonstration successful.")


if __name__ == "__main__":
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    logger = get_logger("demo_script")

    print(f"Running on device: {Config.DEVICE}")

    # 2. Dataset Demo
    train_ds, val_ds = demonstrate_dataset_and_transforms()

    # 3. Model Demo
    demonstrate_models()

    # 4. Engine Demo
    model, train_loader, optimizer = demonstrate_training_engine(train_ds, val_ds)

    # 5. SWA Demo
    demonstrate_swa(model, train_loader, optimizer)

    print("\nAll demonstrations completed successfully.")
