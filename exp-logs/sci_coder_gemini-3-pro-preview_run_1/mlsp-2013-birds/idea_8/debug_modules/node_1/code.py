import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import set_seed, check_tensor_sanitation
from library.dataset import get_data, BirdDataset, get_transforms
from library.model import get_model
from library.swa_utils import SWAHandler, update_bn_statistics
from library.engine import train_one_epoch, validate, predict
from library.pipeline import (
    train_teacher_ensemble,
    generate_sanitized_pseudo_labels,
    train_student_swa,
    generate_submission,
)


def main():
    print("=== Starting Library Usage Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # -------------------------------------------------------------------------
    print("\n[1] Configuring Environment and Overriding Hyperparameters...")

    # Set a fixed seed for reproducibility
    set_seed(42)

    # Define a working directory for this demo
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Monkey-patch the Config class to optimize for speed and use demo paths
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Reduce image size for faster processing
    Config.IMG_HEIGHT = 64
    Config.IMG_WIDTH = 128

    # Reduce training parameters
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    Config.TEACHER_EPOCHS = 1
    Config.STUDENT_EPOCHS = 2
    Config.SWA_START_EPOCH = 1
    Config.NUM_TEACHERS = 1
    Config.DEBUG = True

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Image Size: {Config.IMG_HEIGHT}x{Config.IMG_WIDTH}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading and Processing
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Loading...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Subset metadata for speed (take top 10 samples)
    subset_size = 10
    train_df_sub = train_df.head(subset_size)
    val_df_sub = val_df.head(subset_size)
    test_df_sub = test_df.head(subset_size)

    print(f"    Loading subset of {subset_size} samples for Train/Val/Test...")

    # Use get_data. We disable loading from cache to ensure we process the subset fresh
    # and respect the new image dimensions.
    train_data = get_data(
        train_df_sub, load_cached_data=False, cache_prefix="demo_train"
    )
    val_data = get_data(val_df_sub, load_cached_data=False, cache_prefix="demo_val")
    test_data = get_data(test_df_sub, load_cached_data=False, cache_prefix="demo_test")

    # Unpack
    train_images, train_labels, train_ids = train_data

    # Validations
    assert len(train_images) == subset_size
    assert train_images.shape == (subset_size, Config.IMG_HEIGHT, Config.IMG_WIDTH, 3)
    assert train_labels.shape == (subset_size, Config.NUM_CLASSES)
    assert train_images.dtype == np.uint8
    print("    Data loaded and shapes verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Dataset and DataLoader
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating BirdDataset and DataLoader...")

    # Instantiate Dataset
    dataset = BirdDataset(
        train_images, train_labels, train_ids, transform=get_transforms("train")
    )

    # Check single item
    img_tensor, lbl_tensor, rec_id = dataset[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.dim() == 3  # (C, H, W)
    assert img_tensor.shape[0] == 3
    assert isinstance(lbl_tensor, torch.Tensor)
    assert lbl_tensor.shape[0] == Config.NUM_CLASSES

    # Instantiate DataLoader
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Check batch
    batch_imgs, batch_lbls, batch_ids = next(iter(loader))
    assert batch_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    )
    assert batch_lbls.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)

    print("    Dataset and DataLoader functioning correctly.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Model Initialization...")

    device = Config.DEVICE
    model = get_model(device=device, pretrained=False)  # False for speed

    # Verify model structure
    assert isinstance(model, nn.Module)
    # Check output shape
    dummy_input = torch.randn(2, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, Config.NUM_CLASSES)
    print("    Model initialized and forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Engine: Training and Validation
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Engine (Train/Val)...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train one epoch
    print("    Running train_one_epoch...")
    loss = train_one_epoch(model, optimizer, loader, device, epoch=0)
    assert isinstance(loss, float)
    assert loss >= 0
    print(f"    Training Loss: {loss:.4f}")

    # Validate
    print("    Running validate...")
    val_loss, val_auc = validate(model, loader, device)
    assert isinstance(val_loss, float)
    assert 0.0 <= val_auc <= 1.0
    print(f"    Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # -------------------------------------------------------------------------
    # 6. SWA Handler
    # -------------------------------------------------------------------------
    print("\n[6] Demonstrating SWA Handler...")

    swa_handler = SWAHandler(model)
    swa_model = swa_handler.get_averaged_model()

    # Check that SWA model is a deep copy
    assert swa_model is not model

    # Perform an update
    # Perturb model weights slightly to ensure update changes something
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)

    swa_handler.update(model)

    # Verify BN update function runs
    update_bn_statistics(swa_model, loader, device)
    print("    SWA Handler update and BN statistics update successful.")

    # -------------------------------------------------------------------------
    # 7. Full Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[7] Demonstrating Full Pipeline Execution...")

    # Stage 1: Train Teacher
    # We use the subset data loaded earlier
    print("    >> Stage 1: Training Teacher Ensemble...")
    teachers = train_teacher_ensemble(
        train_data,
        val_data,
        num_teachers=Config.NUM_TEACHERS,
        epochs=Config.TEACHER_EPOCHS,
    )
    assert len(teachers) == Config.NUM_TEACHERS

    # Stage 2: Generate Pseudo Labels
    print("    >> Stage 2: Generating Pseudo-Labels...")
    # We use test_data subset
    pseudo_labels = generate_sanitized_pseudo_labels(
        teachers, test_data, load_cached_data=False
    )
    assert pseudo_labels.shape == (subset_size, Config.NUM_CLASSES)
    check_tensor_sanitation(pseudo_labels, "Demo Pseudo Labels")

    # Stage 3: Train Student with SWA
    print("    >> Stage 3: Training Student with SWA...")
    student_model = train_student_swa(
        train_data, test_data, pseudo_labels, epochs=Config.STUDENT_EPOCHS
    )
    assert isinstance(student_model, nn.Module)

    # Submission
    print("    >> Generating Submission...")
    generate_submission(student_model, test_data)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH)
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Expected rows: subset_size * 19 species
    expected_rows = subset_size * Config.NUM_CLASSES
    assert len(sub_df) == expected_rows
    assert "Id" in sub_df.columns
    assert "Probability" in sub_df.columns

    print(
        f"    Submission generated at {Config.SUBMISSION_PATH} with {len(sub_df)} rows."
    )

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
