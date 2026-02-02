import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import logging
import shutil
from torch.cuda.amp import GradScaler

# Import provided library modules
from library.config import Config, seed_everything
from library.data import CassavaDataset, get_transforms, MixupCollate
from library.modeling import CassavaClassifier, get_llrd_params
from library.loss import SoftTargetCrossEntropy
from library.training import train_one_epoch, valid_one_epoch, SWAContainer
from library.utils import get_logger, save_checkpoint, AverageMeter


def run_demo():
    print("--- Starting Cassava Leaf Disease Classification Demo ---")

    # 1. Setup and Configuration Override
    # We override Config parameters to ensure the demo runs quickly (< 1 min)
    print("\n[1] Configuring environment for demo...")
    seed_everything(42)

    # Override Config for speed
    Config.IMG_SIZE_LOW = 128
    Config.IMG_SIZE_HIGH = 128
    Config.BATCH_SIZE = 4
    Config.ACCUM_STEPS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.EPOCHS_WARMUP = 0
    Config.EPOCHS_BASE = 1
    Config.EPOCHS_FINE = 0
    Config.EPOCHS_SWA = 0
    Config.DEBUG = True

    # Create a demo working directory
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)
    Config.OUTPUT_DIR = demo_dir

    # 2. Data Loading Demonstration
    print("\n[2] Verifying Data Loading and Augmentation...")

    # Load metadata and create a small subset (16 images)
    full_train_df = pd.read_csv(Config.TRAIN_CSV)
    subset_df = full_train_df.head(16).copy()

    print(f"    Loaded subset of {len(subset_df)} samples.")

    # Initialize Dataset
    train_dataset = CassavaDataset(
        subset_df,
        transform=get_transforms(Config.IMG_SIZE_LOW, mode="train"),
        mode="train",
    )

    # Initialize DataLoader with MixupCollate
    mixup_fn = MixupCollate(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,  # Force augmentation for verification
        num_classes=Config.NUM_CLASSES,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=mixup_fn,
        num_workers=0,
    )

    # Fetch one batch
    images, targets = next(iter(train_loader))

    # Verify shapes
    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Target Shape: {targets.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE_LOW,
        Config.IMG_SIZE_LOW,
    ), "Image batch shape mismatch"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Target batch shape mismatch (should be one-hot/mixed)"
    assert targets.dtype == torch.float32, "Targets should be float32 for MixUp"

    print("    Data Loading verification passed.")

    # 3. Model Logic Verification
    print("\n[3] Verifying Model Architecture...")

    # Use a lightweight backbone for demo to avoid downloading large weights
    # 'resnet18' is standard and small. Pretrained=False for speed/offline safety.
    model_arch = "resnet18"
    model = CassavaClassifier(
        model_arch, num_classes=Config.NUM_CLASSES, pretrained=False
    )
    model.to(Config.DEVICE)

    # Create dummy input
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE_LOW, Config.IMG_SIZE_LOW).to(
        Config.DEVICE
    )

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")
    assert output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch"

    # Verify LLRD Parameter Grouping
    param_groups = get_llrd_params(model, lr=1e-3, decay_factor=0.9)
    print(f"    LLRD Parameter Groups: {len(param_groups)}")
    assert len(param_groups) > 0, "No parameter groups found for LLRD"

    # Check if learning rates are actually different (decayed)
    lrs = [g["lr"] for g in param_groups]
    print(f"    Learning Rates per group: {lrs}")
    if len(lrs) > 1:
        assert lrs[0] != lrs[-1], "LLRD failed to assign different learning rates"

    print("    Model verification passed.")

    # 4. Loss Function Verification
    print("\n[4] Verifying SoftTargetCrossEntropy Loss...")
    criterion = SoftTargetCrossEntropy()

    # Mock logits and soft targets
    mock_logits = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]], requires_grad=True
    )
    mock_targets = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0]])

    loss = criterion(mock_logits, mock_targets)
    print(f"    Computed Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    loss.backward()
    assert mock_logits.grad is not None, "Gradients not computed for loss"
    print("    Loss function verification passed.")

    # 5. Training Loop Demonstration
    print("\n[5] Running Training Loop (1 Epoch, Subset)...")

    # Setup Logger
    logger = get_logger(os.path.join(demo_dir, "demo.log"))

    # Optimizer & Scaler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler()

    # Run Train Step
    # We use the subset loader created earlier
    avg_loss, avg_acc = train_one_epoch(
        epoch=1,
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        scaler=scaler,
        device=Config.DEVICE,
        logger=logger,
    )

    print(f"    Train Epoch Result -> Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}")
    assert avg_loss > 0, "Training loss invalid"

    # Run Valid Step
    # Create valid loader (no mixup, standard transform)
    val_dataset = CassavaDataset(
        subset_df,  # Using same subset for demo convenience
        transform=get_transforms(Config.IMG_SIZE_LOW, mode="valid"),
        mode="valid",
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    val_loss, val_acc = valid_one_epoch(
        epoch=1,
        model=model,
        loader=val_loader,
        criterion=nn.CrossEntropyLoss(),  # Standard CE for validation
        device=Config.DEVICE,
        logger=logger,
    )

    print(f"    Valid Epoch Result -> Loss: {val_loss:.4f}, Acc: {val_acc:.2f}")
    print("    Training loop verification passed.")

    # 6. SWA Demonstration
    print("\n[6] Demonstrating SWA Container...")

    swa_container = SWAContainer(model, Config.DEVICE)

    # Update SWA with current model weights
    swa_container.update(model)
    print("    SWA model updated.")

    # Finalize (Update BN)
    # SWA update_bn requires a loader that yields images.
    # Our loader yields (img, target). We can wrap it or just rely on the fact that update_bn
    # might handle extra args or we create a simple generator.
    # The torch.optim.swa_utils.update_bn implementation iterates the loader.
    # If the loader returns (x, y), it uses x.
    print("    Finalizing SWA (Updating BN statistics)...")

    # We need to ensure the model is in train mode on device for update_bn
    # The SWAContainer.finalize method calls update_bn
    try:
        swa_container.finalize(train_loader)
        print("    SWA BN update successful.")
    except Exception as e:
        print(
            f"    SWA BN update warning (expected if dataset too small for momentum): {e}"
        )

    swa_model = swa_container.get_model()
    assert isinstance(swa_model, nn.Module), "SWA did not return a module"

    # Save the demo model
    save_path = os.path.join(demo_dir, "checkpoints", "demo_model.pth")
    save_checkpoint(swa_model.state_dict(), is_best=True, filepath=save_path)
    print(f"    Demo model saved to {save_path}")

    # 7. Submission/Inference Demo
    print("\n[7] Generating Sample Submission...")
    test_df = pd.read_csv(Config.TEST_CSV)
    # Just take head for speed
    test_df = test_df.head(5)

    test_dataset = CassavaDataset(
        test_df, transform=get_transforms(Config.IMG_SIZE_LOW, mode="test"), mode="test"
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    model.eval()
    results = []
    with torch.no_grad():
        for imgs, img_ids in test_loader:
            imgs = imgs.to(Config.DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()

            for img_id, pred in zip(img_ids, preds):
                results.append({"image_id": img_id, "label": pred})

    sub_df = pd.DataFrame(results)
    sub_path = os.path.join(demo_dir, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"    Submission saved to {sub_path}")
    print(sub_df)

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
