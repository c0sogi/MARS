import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_map5
from library.dataset import WhaleDataset, get_class_mapping, get_transforms
from library.model import WhaleDenseNet
from library.loss import WhaleLoss
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== 1. Configuration & Setup ===")
    # Modify Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 64  # Small subset for speed
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.TOTAL_EPOCHS = 2
    Config.SWA_START_EPOCH = 1  # Trigger SWA logic immediately after epoch 1
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")
    print("Configuration complete.\n")

    print("=== 2. Dataset Verification ===")
    # Test Train Dataset
    train_ds = WhaleDataset(mode="train", transform=get_transforms("train"))
    print(f"Train Dataset Length (Debug): {len(train_ds)}")

    # Verify item structure
    img, label, img_id = train_ds[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Label: {label} (Type: {type(label)})")
    print(f"Sample Image ID: {img_id}")

    assert len(train_ds) == Config.DEBUG_SUBSET_SIZE, "Dataset subset size mismatch"
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Image tensor shape mismatch"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"

    # Verify Class Mapping
    classes, class_to_idx = get_class_mapping(load_cached_data=False)
    print(f"Total Unique Classes: {len(classes)}")
    assert len(classes) == Config.NUM_CLASSES, "Class count mismatch with Config"
    assert (
        classes[label.item()] in class_to_idx
    ), "Label mapping consistency check failed"
    print("Dataset verification passed.\n")

    print("=== 3. Model & Loss Verification ===")
    device = Config.DEVICE

    # Instantiate Model
    model = WhaleDenseNet(
        backbone_name=Config.BACKBONE,
        pretrained=False,  # False for speed in demo, usually True
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    # Create dummy batch
    dummy_input = torch.randn(4, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    dummy_labels = torch.tensor([0, 1, 2, 0]).to(device)  # Random labels

    # Forward Pass
    logits = model(dummy_input, dummy_labels)
    print(f"Logits Shape: {logits.shape}")

    assert logits.shape == (4, Config.NUM_CLASSES), "Output shape mismatch"

    # Loss Calculation
    criterion = WhaleLoss()
    loss = criterion(logits, dummy_labels)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print("Model & Loss verification passed.\n")

    print("=== 4. Training Loop Demonstration ===")
    # Initialize Trainer
    trainer = Trainer()

    # Run Training
    # This will run for Config.TOTAL_EPOCHS (2 epochs)
    # It handles training, validation, SWA, and checkpointing
    trainer.fit()

    # Verify Checkpoints
    expected_checkpoint = os.path.join(Config.CHECKPOINT_DIR, "swa_model_final.pth.tar")
    assert os.path.exists(
        expected_checkpoint
    ), f"SWA Checkpoint not found at {expected_checkpoint}"
    print(f"Training completed. Checkpoint saved at: {expected_checkpoint}\n")

    print("=== 5. Inference & Submission Generation ===")
    # Load Test Dataset
    test_ds = WhaleDataset(mode="test", transform=get_transforms("test"))
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Best Model (SWA)
    print("Loading SWA model for inference...")
    checkpoint = torch.load(expected_checkpoint, map_location=device)

    # Note: Trainer wraps model in AveragedModel for SWA.
    # The state_dict keys might have 'module.' prefix if DataParallel was used,
    # or be directly compatible if AveragedModel was saved correctly.
    # In library/trainer.py, it saves swa_model.state_dict().
    # We need to load this into a fresh model instance.

    inference_model = WhaleDenseNet(pretrained=False).to(device)

    # SWA model state dict keys usually match the base model if AveragedModel wraps it directly
    # but AveragedModel keys start with 'module.'
    state_dict = checkpoint["state_dict"]

    # Fix keys if they start with 'module.' (AveragedModel wrapper)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    inference_model.load_state_dict(new_state_dict)
    inference_model.eval()

    # Run Inference
    results = []
    print(f"Running inference on {len(test_ds)} test images...")

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            # TTA: Original + Flip
            logits = inference_model(images, labels=None)  # labels=None for inference

            if Config.TTA_FLIP:
                logits_flip = inference_model(torch.flip(images, dims=[3]), labels=None)
                logits = (logits + logits_flip) / 2.0

            # Get Top 5
            _, top_indices = torch.topk(logits, k=5, dim=1)
            top_indices = top_indices.cpu().numpy()

            # Map indices back to class names
            # We need the index-to-class mapping
            # classes array from get_class_mapping is sorted, so index i -> classes[i]
            batch_preds = []
            for idx_list in top_indices:
                pred_names = [classes[i] for i in idx_list]
                batch_preds.append(" ".join(pred_names))

            for img_id, pred_str in zip(image_ids, batch_preds):
                results.append({"Image": img_id, "Id": pred_str})

    # Create Submission DataFrame
    df_sub = pd.DataFrame(results)
    print(f"Generated {len(df_sub)} predictions.")
    print("First 5 predictions:")
    print(df_sub.head())

    # Save Submission
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
