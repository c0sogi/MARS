import os
import torch
import pandas as pd
import numpy as np
import warnings
import logging
import shutil

# Import from provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import load_metadata, get_class_mapping, DogDataset, get_transforms
from library.model import DogModel
from library.engine import train_fold, predict_tta, save_submission, evaluate
from library.soup import create_greedy_soup

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Library Demo ===\n")

    # ---------------------------------------------------------
    # 1. Setup and Configuration Overrides for Speed
    # ---------------------------------------------------------
    print("[1] Setting up configuration for fast demonstration...")

    # Override Config for speed
    Config.SEED = 42
    Config.EPOCHS = 2
    Config.WARMUP_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.IMG_SIZE = 128  # Reduce image size for speed
    Config.SOUP_CANDIDATES = 2
    # Use a lightweight model for the demo to avoid large downloads/compute
    Config.MODEL_NAME = "resnet18"

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Setup logger
    logger = get_logger("demo_script")
    logger.info("Configuration configured for demo.")

    # ---------------------------------------------------------
    # 2. Data Loading and Subsetting
    # ---------------------------------------------------------
    print("\n[2] Loading and subsetting data...")

    # Load metadata
    df_train_full, df_val_full, df_test_full = load_metadata()

    # Subsample for demo speed (50 train, 20 val, 10 test)
    df_train = df_train_full.iloc[:50].reset_index(drop=True)
    df_val = df_val_full.iloc[:20].reset_index(drop=True)
    df_test = df_test_full.iloc[:10].reset_index(drop=True)

    print(f"    Train subset: {len(df_train)}")
    print(f"    Val subset:   {len(df_val)}")
    print(f"    Test subset:  {len(df_test)}")

    # Get class list
    class_list = get_class_mapping()
    assert len(class_list) > 0, "Class list should not be empty"
    Config.NUM_CLASSES = len(class_list)  # Ensure config matches data

    # ---------------------------------------------------------
    # 3. Dataset and DataLoader Demonstration
    # ---------------------------------------------------------
    print("\n[3] Initializing Datasets and DataLoaders...")

    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    train_dataset = DogDataset(df_train, class_list, transform=train_transforms)
    val_dataset = DogDataset(df_val, class_list, transform=val_transforms)
    test_dataset = DogDataset(df_test, class_list, transform=val_transforms)

    # Validate Dataset item
    img, label = train_dataset[0]
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Unexpected image shape: {img.shape}"
    assert isinstance(label, int), "Label should be an integer"
    assert 0 <= label < Config.NUM_CLASSES, "Label out of bounds"

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    print("    Datasets and Loaders initialized successfully.")

    # ---------------------------------------------------------
    # 4. Model Initialization and Logic Verification
    # ---------------------------------------------------------
    print("\n[4] Initializing Model and verifying logic...")

    device = Config.DEVICE
    model = DogModel(model_name=Config.MODEL_NAME, pretrained=True)
    model.to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch: {output.shape}"

    # Verify Freeze/Unfreeze Logic
    model.freeze_backbone()
    # Check if backbone params are frozen
    for name, param in model.backbone.named_parameters():
        # The classifier head should be unfrozen, others frozen.
        # In timm resnet, the head is usually 'fc'.
        if "fc" not in name and "head" not in name:
            assert (
                param.requires_grad is False
            ), f"Backbone parameter {name} should be frozen"

    model.unfreeze_backbone()
    # Check if all params are unfrozen
    for param in model.parameters():
        assert param.requires_grad is True, "All parameters should be unfrozen"

    print("    Model logic (forward, freeze, unfreeze) verified.")

    # ---------------------------------------------------------
    # 5. Training Engine Demonstration (Train Fold)
    # ---------------------------------------------------------
    print("\n[5] Running Training Engine (Fold 0)...")

    # We re-initialize model to start fresh
    model = DogModel(model_name=Config.MODEL_NAME, pretrained=True)
    model.to(device)

    # train_fold handles Warmup -> Fine-tuning -> In-memory Soup
    trained_model = train_fold(model, train_loader, val_loader, device, fold_idx=0)

    # Validate that the model is returned and works
    val_loss = evaluate(trained_model, val_loader, device)
    print(f"    Post-training Validation Loss: {val_loss:.4f}")
    assert val_loss > 0, "Validation loss should be positive"

    # ---------------------------------------------------------
    # 6. Disk-Based Soup Demonstration
    # ---------------------------------------------------------
    print("\n[6] Demonstrating Disk-Based Greedy Model Soup...")

    # Create dummy checkpoints to simulate saved epochs
    ckpt_dir = os.path.join(Config.WORKING_DIR, "demo_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    path1 = os.path.join(ckpt_dir, "ckpt_1.pth")
    path2 = os.path.join(ckpt_dir, "ckpt_2.pth")

    # Save current model as ckpt1
    torch.save(trained_model.state_dict(), path1)

    # Modify model slightly and save as ckpt2
    with torch.no_grad():
        for param in trained_model.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    torch.save(trained_model.state_dict(), path2)

    # Run soup creation from disk
    soup_state = create_greedy_soup(
        model=DogModel(
            model_name=Config.MODEL_NAME, pretrained=False
        ),  # Empty model structure
        loader=val_loader,
        checkpoint_paths=[path1, path2],
        device=device,
    )

    assert soup_state is not None, "Soup creation returned None"
    # Load soup into model
    trained_model.load_state_dict(soup_state)
    print("    Disk-based soup creation successful.")

    # ---------------------------------------------------------
    # 7. Inference and Submission
    # ---------------------------------------------------------
    print("\n[7] Running Inference (TTA) and generating submission...")

    preds = predict_tta(trained_model, test_loader, device)

    assert preds.shape == (
        len(df_test),
        Config.NUM_CLASSES,
    ), f"Prediction shape mismatch. Expected ({len(df_test)}, {Config.NUM_CLASSES}), got {preds.shape}"

    # Create submission file
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    breed_cols = df_test.columns.drop(["id", "file_path"]).tolist()

    # In the test metadata provided, the breed columns are placeholders.
    # We need the actual class names from our class_list mapping to ensure correct headers.
    # The sample submission format requires headers: id, breed1, breed2...
    # The class_list from get_class_mapping() is sorted and matches the model output indices.

    save_submission(preds, df_test["id"].tolist(), class_list, submission_path)

    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify file content
    df_sub = pd.read_csv(submission_path)
    assert df_sub.shape == (
        len(df_test),
        Config.NUM_CLASSES + 1,
    ), "Submission file has incorrect shape"
    assert "id" in df_sub.columns, "Submission file missing 'id' column"

    print(f"    Submission saved to {submission_path}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
