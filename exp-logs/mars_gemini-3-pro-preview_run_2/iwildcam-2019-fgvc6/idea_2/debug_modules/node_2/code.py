import os
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader
import torch.nn as nn

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_transforms, compute_class_weights
from library.dataset import load_dataset
from library.model import HybridResNet
from library.trainer import (
    train_stage1_warmup,
    train_stage2_finetune,
    generate_submission,
)


def run_demo():
    print("=== Starting Library Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # ---------------------------------------------------------
    print("Configuring environment for rapid demonstration...")

    # Set a fixed seed for reproducibility
    seed_everything(42)

    # Override Config parameters to use a tiny subset and run quickly
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 images
    Config.BATCH_SIZE = 4  # Small batch size
    Config.STAGE1_EPOCHS = 1  # 1 Epoch for warmup
    Config.STAGE2_EPOCHS = 1  # 1 Epoch for fine-tuning
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Use a specific demo directory for outputs
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Re-run setup to create new directories
    Config.setup()

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # 2. Data Loading and Transformation Verification
    # ---------------------------------------------------------
    print("\n--- Verifying Data Loading ---")

    # Get transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")
    test_transform = get_transforms("test")

    # Load datasets (this uses Config.DEBUG_SAMPLE_SIZE)
    train_dataset = load_dataset("train", transform=train_transform)
    val_dataset = load_dataset("val", transform=val_transform)
    test_dataset = load_dataset("test", transform=test_transform)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Assertions
    assert len(train_dataset) == Config.DEBUG_SAMPLE_SIZE, "Train dataset size mismatch"
    assert len(val_dataset) == Config.DEBUG_SAMPLE_SIZE, "Val dataset size mismatch"
    # Test dataset might be smaller if the source file is small, but here we assume it respects debug size

    # Verify item retrieval
    img, label = train_dataset[0]
    assert img.shape == (3, 224, 224), f"Incorrect image shape: {img.shape}"
    assert isinstance(label, torch.Tensor), "Label is not a tensor"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ---------------------------------------------------------
    # 3. Class Weights Verification
    # ---------------------------------------------------------
    print("\n--- Verifying Class Weights ---")
    class_weights = compute_class_weights(train_dataset.df)
    assert len(class_weights) == Config.NUM_CLASSES, "Class weights dimension mismatch"
    assert not torch.isnan(class_weights).any(), "Class weights contain NaNs"
    print("Class weights computed successfully.")

    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---------------------------------------------------------
    # 4. Model Architecture and Freezing Logic Verification
    # ---------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")
    model = HybridResNet()
    model.to(device)

    # Check output shape
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch: {output.shape}"

    # Check Freeze Backbone (Stage 1 logic)
    model.freeze_backbone()
    # Verify backbone parameters are frozen
    for name, param in model.backbone.named_parameters():
        assert (
            param.requires_grad is False
        ), f"Backbone parameter {name} should be frozen"
    # Verify head is trainable
    for name, param in model.fc.named_parameters():
        assert param.requires_grad is True, f"Head parameter {name} should be trainable"
    print("Model freezing logic verified.")

    # ---------------------------------------------------------
    # 5. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n--- Running Training Stages ---")

    # Stage 1: Warmup
    # We use the imported function directly. It handles the loop and validation.
    model = train_stage1_warmup(
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        epochs=Config.STAGE1_EPOCHS,
        lr=0.01,  # Use smaller LR for stability in demo
    )

    # Check Unfreeze Layer 4 (Stage 2 logic)
    # The trainer function calls unfreeze_layer4 internally, but let's verify the method itself works
    # before calling stage 2.
    model.unfreeze_layer4()

    # Verify Layer 4 is unfrozen
    # Layer 4 is the last child in the backbone list (index 7)
    layer4_params = list(model.backbone[7].parameters())
    assert len(layer4_params) > 0, "Layer 4 has no parameters"
    for param in layer4_params:
        assert param.requires_grad is True, "Layer 4 parameters should be trainable"

    # Verify earlier layers (e.g., Layer 1) are still frozen
    layer1_params = list(model.backbone[4].parameters())
    for param in layer1_params:
        assert param.requires_grad is False, "Layer 1 parameters should be frozen"
    print("Model unfreezing logic verified.")

    # Stage 2: Fine-Tuning
    model = train_stage2_finetune(
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        epochs=Config.STAGE2_EPOCHS,
        lr=1e-4,
        patience=1,
    )

    # Verify checkpoint creation
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not saved"
    print("Training pipeline completed successfully.")

    # ---------------------------------------------------------
    # 6. Submission Generation Verification
    # ---------------------------------------------------------
    print("\n--- Generating Submission ---")
    generate_submission(model, test_loader, device, output_path=Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(sub_df.columns) == ["Id", "Predicted"], "Submission columns mismatch"
    assert len(sub_df) == len(test_dataset), "Submission row count mismatch"
    assert sub_df["Predicted"].dtype in [
        int,
        np.int64,
    ], "Predicted column is not integer"

    print("\n=== Demonstration Completed Successfully ===")
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    run_demo()
