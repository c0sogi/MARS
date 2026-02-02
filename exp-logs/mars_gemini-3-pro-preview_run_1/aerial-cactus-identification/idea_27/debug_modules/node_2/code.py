import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders, mixup_data
from library.models import CactusRepVGG, CactusResNet
from library.engine import fit, validate


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print(">>> Setting up configuration for demo run...")

    # Override Config for speed and isolation
    Config.DEBUG = True  # Use small subset of data
    Config.BATCH_SIZE = 16  # Reduce batch size for small debug dataset
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.SWA_START_EPOCH = 1  # Start SWA immediately to test logic
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Initialize environment (creates dirs, sets seeds)
    Config.initialize()
    seed_everything(Config.SEED)

    logger = get_logger(
        name="Demo", log_file=os.path.join(Config.WORKING_DIR, "demo.log")
    )
    logger.info("Configuration initialized.")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    logger.info(">>> Testing Data Pipeline...")

    # Get dataloaders (force reload to ensure logic runs)
    train_loader, val_loader, test_loader, test_ids, quality_range = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Verify Train Loader
    assert len(train_loader) > 0, "Train loader is empty"
    batch = next(iter(train_loader))

    # Check batch structure
    assert "image" in batch
    assert "label" in batch
    assert "quality" in batch

    images = batch["image"]
    labels = batch["label"]
    qualities = batch["quality"]

    # Check dimensions
    # Batch size might be smaller if drop_last=False (though it is True for train)

    # Re-create loaders with new batch size (testing cache loading)
    train_loader, val_loader, test_loader, test_ids, quality_range = get_dataloaders(
        load_cached_data=True, debug=True  # Use cache now
    )

    batch = next(iter(train_loader))
    images = batch["image"]

    assert images.shape[1:] == (3, 32, 32), f"Unexpected image shape: {images.shape}"
    assert batch["label"].shape[0] == images.shape[0], "Label batch size mismatch"

    logger.info(f"Data Loaded Successfully. Batch Shape: {images.shape}")

    # Verify Mixup
    logger.info(">>> Testing Mixup Augmentation...")
    mixed_x, y_a, y_b, q_a, q_b, lam = mixup_data(
        images, batch["label"], batch["quality"], alpha=0.2, device="cpu"
    )
    assert mixed_x.shape == images.shape, "Mixup altered image shape"
    assert 0 <= lam <= 1, "Mixup lambda out of range"
    logger.info("Mixup verification passed.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    logger.info(">>> Testing Model Architectures...")

    device = Config.DEVICE

    # Test ResNet
    resnet = CactusResNet(num_classes=1).to(device)
    with torch.no_grad():
        out = resnet(images.to(device))
        assert "class" in out and "quality" in out
        assert out["class"].shape == (images.shape[0], 1)
    logger.info("ResNet forward pass successful.")

    # Test RepVGG and Re-parameterization
    repvgg = CactusRepVGG(num_classes=1, deploy=False).to(device)
    repvgg.eval()

    # Save state before deploy
    input_tensor = torch.randn(2, 3, 32, 32).to(device)
    with torch.no_grad():
        out_train_mode = repvgg(input_tensor)["class"]

    # Switch to deploy
    logger.info("Testing RepVGG switch_to_deploy...")
    repvgg.switch_to_deploy()

    with torch.no_grad():
        out_deploy_mode = repvgg(input_tensor)["class"]

    # Check consistency (allow small tolerance for float precision)
    diff = (out_train_mode - out_deploy_mode).abs().max().item()
    logger.info(f"RepVGG mode difference: {diff:.6f}")
    assert diff < 1e-4, "RepVGG re-parameterization failed (outputs diverge)"
    logger.info("RepVGG re-parameterization verified.")

    # --------------------------------------------------------------------------
    # 4. Training Engine Verification
    # --------------------------------------------------------------------------
    logger.info(">>> Testing Training Loop (Engine)...")

    # Re-instantiate model for training
    model = CactusRepVGG(num_classes=1, deploy=False).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # Run Fit
    best_auc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=2,
    )

    logger.info(f"Training finished. Best AUC: {best_auc:.4f}")
    assert 0 <= best_auc <= 1, "Invalid AUC score returned"

    # --------------------------------------------------------------------------
    # 5. Checkpoint & Output Verification
    # --------------------------------------------------------------------------
    logger.info(">>> Verifying Outputs...")

    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    swa_model_path = os.path.join(Config.CHECKPOINT_DIR, "swa_model.pth")

    assert os.path.exists(best_model_path), "best_model.pth not found"
    assert os.path.exists(swa_model_path), "swa_model.pth not found"

    logger.info("Checkpoints found.")

    # --------------------------------------------------------------------------
    # 6. Inference Demonstration
    # --------------------------------------------------------------------------
    logger.info(">>> Demonstrating Inference...")

    # Load SWA model
    inference_model = CactusRepVGG(num_classes=1, deploy=False)
    state_dict = torch.load(swa_model_path, map_location=device)
    inference_model.load_state_dict(state_dict)
    inference_model.to(device)
    inference_model.switch_to_deploy()  # Optimize for inference
    inference_model.eval()

    # Run inference on test set
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            out = inference_model(imgs)
            probs = torch.sigmoid(out["class"]).cpu().numpy().flatten()
            preds.extend(probs)

    preds = np.array(preds)

    # Create submission file
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": preds})

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(sub_path, index=False)

    logger.info(f"Submission generated at {sub_path} with shape {submission_df.shape}")
    assert len(submission_df) == len(test_ids), "Submission length mismatch"

    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
