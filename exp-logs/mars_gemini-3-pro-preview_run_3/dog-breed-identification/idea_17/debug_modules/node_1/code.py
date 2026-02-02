import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil

# Import provided library components
from library.config import Config
from library.utils import seed_everything, generate_model_soup
from library.dataset import get_dataloaders, get_metadata
from library.model import BreedClassifier
from library.engine import train_one_epoch, evaluate, predict_tta

if __name__ == "__main__":
    # ==========================================
    # 1. Setup and Configuration Override
    # ==========================================
    print("Initializing demonstration...")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Override Config for rapid demonstration
    # We use DEBUG mode to limit dataset size to a small subset
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = (
        50  # Small enough for speed, large enough for a few batches
    )
    Config.BATCH_SIZE = 8  # Small batch size for the demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.EPOCHS_WARMUP = 1
    Config.EPOCHS_FINE = 1

    # Ensure working directory is clean/ready
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n--- Step 1: Loading Data ---")

    # Get dataloaders
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        load_cached_data=False,  # Force regeneration of cache for demo
        batch_size=Config.BATCH_SIZE,
    )

    print(f"Number of classes: {len(classes)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Assertion checks
    assert len(classes) == 120, f"Expected 120 classes, got {len(classes)}"

    # Check one batch structure
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image tensor shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label tensor shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.long, "Labels should be long (int64)"

    # ==========================================
    # 3. Model Initialization & Logic Check
    # ==========================================
    print("\n--- Step 2: Model Initialization ---")

    device = Config.DEVICE
    print(f"Using device: {device}")

    model = BreedClassifier(num_classes=len(classes), pretrained=True)
    model.to(device)

    # Verify Freeze Logic
    model.freeze_backbone()

    # Check if backbone params are frozen (requires_grad=False)
    # We check a parameter from the backbone (e.g., stem or first block)
    backbone_param = next(model.backbone.parameters())
    assert backbone_param.requires_grad is False, "Backbone should be frozen"

    # Check if head is unfrozen
    # The head parameters are usually the last ones
    head_params = list(model.backbone.get_classifier().parameters())
    assert len(head_params) > 0, "Classifier head not found"
    assert head_params[0].requires_grad is True, "Classifier head should be trainable"

    print("Model freeze logic verified.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("\n--- Step 3: Training Loop (Warmup Phase) ---")

    # Optimizer for warmup (only head params effectively updated)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LR_WARMUP
    )

    # Train 1 epoch
    loss, acc = train_one_epoch(model, train_loader, optimizer, device, epoch=1)
    print(f"Warmup - Loss: {loss:.4f}, Acc: {acc:.4f}")

    assert not np.isnan(loss), "Training loss is NaN"

    # Save checkpoint 1 for soup demo
    ckpt1_path = os.path.join(Config.WORKING_DIR, "ckpt_epoch_1.pth")
    torch.save(model.state_dict(), ckpt1_path)

    print("\n--- Step 4: Training Loop (Fine-Tuning Phase) ---")

    # Unfreeze backbone
    model.unfreeze_backbone()
    backbone_param = next(model.backbone.parameters())
    assert backbone_param.requires_grad is True, "Backbone should be unfrozen"

    # Optimizer for fine-tuning (all params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR_FINE)

    # Train 1 epoch
    loss, acc = train_one_epoch(model, train_loader, optimizer, device, epoch=2)
    print(f"Fine-tune - Loss: {loss:.4f}, Acc: {acc:.4f}")

    # Evaluate
    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Save checkpoint 2 for soup demo
    ckpt2_path = os.path.join(Config.WORKING_DIR, "ckpt_epoch_2.pth")
    torch.save(model.state_dict(), ckpt2_path)

    # ==========================================
    # 5. Model Soup Demonstration
    # ==========================================
    print("\n--- Step 5: Model Soup Generation ---")

    soup_path = os.path.join(Config.WORKING_DIR, "best_soup.pth")

    # Generate soup from the two checkpoints
    generate_model_soup([ckpt1_path, ckpt2_path], soup_path)

    assert os.path.exists(soup_path), "Soup model file was not created"

    # Verify we can load the soup
    soup_state = torch.load(soup_path, map_location=device)
    model.load_state_dict(soup_state)
    print("Model soup loaded successfully.")

    # ==========================================
    # 6. Inference (TTA) & Submission
    # ==========================================
    print("\n--- Step 6: Inference and Submission ---")

    # Predict using TTA
    probs, ids = predict_tta(model, test_loader, device)

    print(f"Predictions shape: {probs.shape}")
    print(f"Number of IDs: {len(ids)}")

    assert len(probs) == len(ids), "Mismatch between predictions and IDs"
    assert (
        probs.shape[1] == 120
    ), f"Expected 120 class probabilities, got {probs.shape[1]}"

    # Create submission DataFrame
    # Columns must be: id, breed1, breed2, ...
    submission_df = pd.DataFrame(probs, columns=classes)
    submission_df.insert(0, "id", ids)

    # Verify submission format
    print("Sample submission rows:")
    print(submission_df.head(2))

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    print("\nDemonstration completed successfully.")
