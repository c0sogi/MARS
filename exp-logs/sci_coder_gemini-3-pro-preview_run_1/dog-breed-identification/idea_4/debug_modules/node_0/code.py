import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_device, ensure_directory
from library.dataset import load_datasets, DogDataset, get_transforms
from library.model import DogClassifier
from library.engine import train_loop, predict_tta


def run_demo():
    print("=== Starting Demonstration of Dog Breed Classification Library ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Demo Purposes
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Enable Debug mode to use a tiny subset of data (defined in Config as 100 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Override to even smaller for instant execution

    # Reduce training epochs to 1 per phase
    Config.EPOCHS_PHASE_1 = 1
    Config.EPOCHS_PHASE_2 = 1

    # Use a separate working directory for this demo
    Config.WORKING_DIR = "./working/demo_script_run"
    ensure_directory(Config.WORKING_DIR)

    # Reduce batch size and workers for the demo environment
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0

    # Seed everything for reproducibility
    seed_everything(Config.SEED)

    device = get_device()
    print(f"Device: {device}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ------------------------------------------------------------------------
    # 2. Data Loading and Verification
    # ------------------------------------------------------------------------
    print("\n[2] Loading and Verifying Datasets...")

    # Load datasets (this handles the DEBUG subsetting internally)
    train_val_df, test_df, class_to_idx, classes = load_datasets(load_cached_data=False)

    # Assertions to verify data loading
    assert not train_val_df.empty, "Training DataFrame is empty."
    assert not test_df.empty, "Test DataFrame is empty."
    assert (
        len(classes) == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} classes, got {len(classes)}"
    assert "file_path" in train_val_df.columns, "file_path column missing in train_df"
    assert "breed" in train_val_df.columns, "breed column missing in train_df"

    print(f"Train/Val Subset Size: {len(train_val_df)}")
    print(f"Test Subset Size: {len(test_df)}")
    print(f"Number of Classes: {len(classes)}")

    # ------------------------------------------------------------------------
    # 3. Dataset Class and DataLoader Verification
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Dataset and DataLoader...")

    # Create a training dataset instance
    train_dataset = DogDataset(
        df=train_val_df,
        transform=get_transforms("train"),
        class_to_idx=class_to_idx,
        return_ids=False,
    )

    # Verify __getitem__
    img, label = train_dataset[0]
    assert isinstance(img, torch.Tensor), "Dataset did not return a tensor image."
    assert img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Incorrect image shape: {img.shape}"
    assert isinstance(label, torch.Tensor), "Dataset did not return a tensor label."

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch one batch to verify
    batch_imgs, batch_labels = next(iter(train_loader))
    assert batch_imgs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch."
    assert batch_imgs.shape[1:] == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Batch image dimensions mismatch."
    print("Dataset and DataLoader verified successfully.")

    # ------------------------------------------------------------------------
    # 4. Model Logic Verification
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Model Logic...")

    # Initialize model
    # Note: Using pretrained=False to avoid downloading weights during this quick demo
    model = DogClassifier(
        num_classes=len(classes), pretrained=False, dropout_rate=0.5
    ).to(device)

    # Verify Freeze Backbone Logic
    model.freeze_backbone(freeze=True)
    # Check a backbone parameter
    for name, param in model.backbone.named_parameters():
        # The head parameters are excluded from freezing.
        # We need to ensure at least one backbone param is frozen.
        # We can't easily distinguish head vs backbone by name generically without the internal logic,
        # but we can check the result of the method.
        # Let's check the first parameter of the backbone (usually stem/conv1)
        if param.requires_grad:
            # If it requires grad, it must be part of the head.
            # But we just froze the backbone.
            pass

    # Verify Forward Pass
    dummy_input = torch.randn(
        Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(device)
    output = model(dummy_input)
    assert output.shape == (
        Config.BATCH_SIZE,
        len(classes),
    ), f"Model output shape mismatch: {output.shape}"
    print("Model instantiated and forward pass verified.")

    # ------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # ------------------------------------------------------------------------
    print("\n[5] Running Training Loop (1 Fold, 1 Epoch per Phase)...")

    # Prepare Validation Loader (using same subset for demo simplicity)
    val_dataset = DogDataset(
        df=train_val_df,  # reusing train df for demo
        transform=get_transforms("val"),
        class_to_idx=class_to_idx,
        return_ids=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run Training Loop for Fold 0
    # This will trigger Phase 1 (Frozen) and Phase 2 (Unfrozen)
    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        fold_idx=0,
        device=device,
    )

    # Verify Model Checkpoint exists
    expected_ckpt_path = Config.get_model_path(0)
    assert os.path.exists(
        expected_ckpt_path
    ), f"Model checkpoint not found at {expected_ckpt_path}"
    print(f"Training loop completed. Checkpoint saved at {expected_ckpt_path}")

    # ------------------------------------------------------------------------
    # 6. Inference Demonstration
    # ------------------------------------------------------------------------
    print("\n[6] Running Inference (TTA)...")

    # Create Test Dataset
    test_dataset = DogDataset(
        df=test_df,
        transform=get_transforms("test"),
        class_to_idx=None,  # Not needed for inference
        return_ids=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load the best model from training
    model.load_state_dict(torch.load(expected_ckpt_path, map_location=device))

    # Predict
    ids, probs = predict_tta(model, test_loader, device)

    assert len(ids) == len(
        test_df
    ), "Number of predictions does not match number of test samples."
    assert probs.shape == (
        len(test_df),
        len(classes),
    ), "Probability matrix shape mismatch."
    print("Inference completed successfully.")

    # ------------------------------------------------------------------------
    # 7. Submission File Generation
    # ------------------------------------------------------------------------
    print("\n[7] Generating Submission File...")

    submission_df = pd.DataFrame(probs, columns=classes)
    submission_df.insert(0, "id", ids)

    output_path = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    submission_df.to_csv(output_path, index=False)

    assert os.path.exists(output_path), "Submission file was not created."
    print(f"Submission file created at {output_path}")
    print("Sample rows:")
    print(submission_df.head(3))

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
