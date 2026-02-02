import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import Library Modules
from library.config import Config
from library.utils import seed_everything, calculate_auc, write_submission
from library.data import get_dataloaders, get_test_dataloader
from library.models import get_model
from library.engine import train_one_epoch, evaluate, SWAManager


def run_demo():
    # 1. Configure Global Settings for Fast Demonstration
    print("Configuring environment for demo...")
    Config.DEBUG = True
    Config.EPOCHS = 2
    Config.PHYSICAL_BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.SWA_START_EPOCH_PCT = 0.5  # Start SWA at epoch 1 (0-indexed logic in loop)

    # Redirect output paths to a demo directory
    Config.PROJECT_NAME = "demo_execution"
    Config.OUTPUT_ROOT = "./working"
    Config.SUBMISSION_ROOT = os.path.join(
        Config.OUTPUT_ROOT, Config.PROJECT_NAME, "submission"
    )
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_ROOT, "submission.csv")

    # Re-initialize paths based on new project name
    Config.IDEA_DIR = os.path.join(Config.OUTPUT_ROOT, Config.PROJECT_NAME)
    Config.CACHE_DIR = os.path.join(Config.IDEA_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.IDEA_DIR, "checkpoints")

    # Manually create directories (usually done in Config.__init__)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_ROOT, exist_ok=True)

    # 2. Initialization
    print("Initializing...")
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 3. Data Loading
    print("Loading Data...")
    # Force load_cached_data=False to verify raw data processing logic
    train_loader, val_loader = get_dataloaders(fold=0, load_cached_data=False)

    # Verify Data Shapes
    images, labels = next(iter(train_loader))
    print(f"Batch Shapes - Images: {images.shape}, Labels: {labels.shape}")

    assert images.shape == (
        Config.PHYSICAL_BATCH_SIZE,
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), "Incorrect image batch shape"
    assert labels.shape == (
        Config.PHYSICAL_BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect label batch shape"

    # 4. Model Setup
    print("Setting up Model...")
    model = get_model("resnet18", pretrained=True)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )
    criterion = nn.BCEWithLogitsLoss()

    swa_manager = SWAManager(model, optimizer)

    # 5. Training Loop
    print("Starting Training Loop...")
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # SWA Step
        swa_manager.step(epoch, model, scheduler)

    # 6. SWA Finalization
    print("\nFinalizing SWA Model...")
    swa_model = swa_manager.finalize(train_loader, device)

    # Verify SWA Model works
    val_loss_swa, val_auc_swa = evaluate(swa_model, val_loader, criterion, device)
    print(f"SWA Val Loss: {val_loss_swa:.4f}, SWA Val AUC: {val_auc_swa:.4f}")

    # 7. Inference
    print("\nRunning Inference on Test Set...")
    test_loader, test_ids = get_test_dataloader(load_cached_data=False)

    swa_model.eval()
    all_probs = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = swa_model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)

    assert len(all_probs) == len(test_ids), "Mismatch between predictions and test IDs"

    # 8. Submission Generation
    print("Generating Submission...")
    submission_ids = []
    submission_probs = []

    # Expand predictions: Rec_ID + Species Index -> ID
    for i, rec_id in enumerate(test_ids):
        rec_probs = all_probs[i]
        for species_idx in range(Config.NUM_CLASSES):
            # Format: rec_id * 100 + species_idx
            # Note: rec_id is integer from metadata
            combined_id = int(rec_id) * 100 + species_idx
            prob = rec_probs[species_idx]

            submission_ids.append(combined_id)
            submission_probs.append(prob)

    # Write Submission
    write_submission(submission_ids, submission_probs)

    # Verify File Creation
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    # Verify Submission Content
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission Head:\n{df_sub.head()}")
    assert (
        len(df_sub) == len(test_ids) * Config.NUM_CLASSES
    ), "Incorrect submission length"
    assert (
        "Id" in df_sub.columns and "Probability" in df_sub.columns
    ), "Missing columns in submission"

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    run_demo()
