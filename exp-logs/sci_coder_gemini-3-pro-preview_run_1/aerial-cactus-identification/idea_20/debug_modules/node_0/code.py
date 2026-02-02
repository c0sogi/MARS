import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.utils import seed_everything, get_device
from library.dataset import load_and_cache_images, CactusDataset, get_transforms
from library.model import RepVGGClassifier, model_to_deploy
from library.engine import train_one_epoch, evaluate, SWAHandler


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # Define paths
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/demo_run"

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    # 2. Data Loading (Subset for Speed)
    print("\n--- Testing Data Loading ---")
    # We load a small subset (max_samples=100) to make this fast
    train_imgs, train_labels, test_imgs, test_ids = load_and_cache_images(
        input_dir=input_dir,
        metadata_dir=metadata_dir,
        cache_dir=os.path.join(working_dir, "cache"),
        load_cached_data=False,  # Force reload to demonstrate loading logic
        max_samples=100,
    )

    print(f"Loaded Train Images: {train_imgs.shape}")
    print(f"Loaded Train Labels: {train_labels.shape}")
    print(f"Loaded Test Images: {test_imgs.shape}")

    # Assertions to verify data loading
    assert train_imgs.shape[0] == 100
    assert train_imgs.shape[1:] == (32, 32, 3)
    assert len(train_labels) == 100

    # 3. Dataset and DataLoader
    print("\n--- Testing Dataset and DataLoader ---")
    # Split subset into train/val
    val_size = 20
    train_x, val_x = train_imgs[:-val_size], train_imgs[-val_size:]
    train_y, val_y = train_labels[:-val_size], train_labels[-val_size:]

    # Create Datasets
    # get_transforms('train') returns True (enable aug), 'val' returns False
    train_ds = CactusDataset(train_x, train_y, transform=get_transforms("train"))
    val_ds = CactusDataset(val_x, val_y, transform=get_transforms("val"))
    test_ds = CactusDataset(test_imgs, transform=get_transforms("test"))

    # Create Loaders
    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Verify Batch Shapes
    sample_imgs, sample_labels = next(iter(train_loader))
    print(f"Batch Image Shape: {sample_imgs.shape}")  # Should be [B, 3, 32, 32]
    print(f"Batch Label Shape: {sample_labels.shape}")

    assert sample_imgs.shape == (batch_size, 3, 32, 32)
    assert sample_imgs.dtype == torch.float32

    # 4. Model Initialization
    print("\n--- Testing Model Initialization ---")
    # Use small width multiplier for speed in this demo
    model = RepVGGClassifier(
        num_classes=1, width_multiplier=[0.5, 0.5, 0.5, 1.0], deploy=False
    )
    model.to(device)

    # Check if model has auxiliary head (since deploy=False)
    assert hasattr(model, "aux_head")
    print("Model initialized successfully (Training Mode).")

    # 5. Training Loop & SWA
    print("\n--- Testing Training Loop & SWA ---")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Initialize SWA Handler (Start SWA immediately at epoch 1 for demo)
    swa_handler = SWAHandler(model, optimizer, swa_start_epoch=1, swa_lr=1e-3)

    epochs = 2
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Train
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, mixup_alpha=0.2
        )
        print(f"  Train Loss: {avg_loss:.4f}")

        # SWA Step
        swa_handler.step(epoch, model)

        # Evaluate
        auc = evaluate(model, val_loader, device)
        print(f"  Val AUC: {auc:.4f}")

        # Basic assertion that loss is valid
        assert not np.isnan(avg_loss)

    # Finalize SWA (Update BN)
    print("Finalizing SWA...")
    swa_model = swa_handler.finalize(train_loader, device)

    # Evaluate SWA model
    swa_auc = evaluate(swa_model, val_loader, device)
    print(f"SWA Val AUC: {swa_auc:.4f}")

    # 6. Model Deployment (RepVGG Conversion)
    print("\n--- Testing Model Deployment (Reparameterization) ---")
    # Convert the SWA model (which wraps the module) to deploy mode
    # swa_model is AveragedModel, so we access .module
    deploy_model = model_to_deploy(swa_model.module)
    deploy_model.to(device)
    deploy_model.eval()

    # Verify structure: deploy model should NOT have 'rbr_dense' or 'aux_head'
    # It should have 'rbr_reparam'
    has_reparam = False
    for name, module in deploy_model.named_modules():
        if hasattr(module, "rbr_reparam"):
            has_reparam = True
        if hasattr(module, "rbr_dense"):
            raise AssertionError("Deploy model still has rbr_dense branch!")

    assert has_reparam, "Deploy model does not seem to have re-parameterized blocks."
    print("Model successfully converted to Deploy Mode.")

    # 7. Inference on Test Set
    print("\n--- Testing Inference with Deployed Model ---")
    preds = []
    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(device)
            # Deploy model returns logits directly
            out = deploy_model(imgs)
            probs = torch.sigmoid(out)
            preds.append(probs.cpu().numpy())

    preds = np.concatenate(preds).flatten()
    print(f"Predictions shape: {preds.shape}")
    print(f"First 5 predictions: {preds[:5]}")

    assert len(preds) == len(test_imgs)
    assert (preds >= 0).all() and (preds <= 1).all()

    # 8. Cleanup
    print("\n--- Cleanup ---")
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
        print(f"Removed {working_dir}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
