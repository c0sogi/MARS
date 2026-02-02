import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Import library components
from library.config import (
    WORKING_DIR,
    DEVICE,
    IMG_SIZE_TEACHER,
    IMG_SIZE_STUDENT,
    BATCH_SIZE,
    NUM_WORKERS,
    SUBMISSION_PATH,
    MODEL_RESNET,
    MODEL_CONVNEXT,
    MODEL_MAXVIT,
)
from library.utils import set_seed, AverageMeter, save_checkpoint
from library.dataset import CatDogDataset, DualResolutionDataset
from library.models import create_model
from library.engine import train_one_epoch, train_distill_one_epoch, validate
from library.inference import inference_fn


def main():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup
    set_seed(42)
    print(f"Device: {DEVICE}")
    print(f"Working Directory: {WORKING_DIR}")

    # 2. Dataset Demonstration & Verification
    print("\n--- Verifying Datasets ---")

    # Standard Dataset (Train split, Debug mode for speed)
    train_ds = CatDogDataset(split="train", debug=True, img_size=IMG_SIZE_TEACHER)
    print(f"Train Dataset (Debug) Size: {len(train_ds)}")
    assert len(train_ds) == 200, "Debug dataset should have 200 samples"

    img, target = train_ds[0]
    assert img.shape == (
        3,
        IMG_SIZE_TEACHER,
        IMG_SIZE_TEACHER,
    ), f"Expected shape (3, {IMG_SIZE_TEACHER}, {IMG_SIZE_TEACHER}), got {img.shape}"
    assert isinstance(target, torch.Tensor), "Target should be a tensor"

    # Dual Resolution Dataset (for Distillation)
    dual_ds = DualResolutionDataset(split="train", debug=True)
    img_t, img_s, target_d = dual_ds[0]
    assert img_t.shape == (
        3,
        IMG_SIZE_TEACHER,
        IMG_SIZE_TEACHER,
    ), "Teacher image shape mismatch"
    assert img_s.shape == (
        3,
        IMG_SIZE_STUDENT,
        IMG_SIZE_STUDENT,
    ), "Student image shape mismatch"
    print("Dataset shapes verified.")

    # 3. Model Initialization
    print("\n--- Initializing Models ---")
    # We use pretrained=False to avoid downloading heavy weights for this demo
    resnet = create_model(MODEL_RESNET, pretrained=False, num_classes=1).to(DEVICE)
    convnext = create_model(MODEL_CONVNEXT, pretrained=False, num_classes=1).to(DEVICE)
    maxvit = create_model(MODEL_MAXVIT, pretrained=False, num_classes=1).to(DEVICE)
    print("Models initialized successfully.")

    # 4. Training Simulation (to generate checkpoints)
    print("\n--- Simulating Training (Generating Checkpoints) ---")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    dual_loader = DataLoader(
        dual_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    # A. Train Teacher 1: ResNet
    print("Training ResNet (Teacher 1)...")
    optimizer_resnet = optim.AdamW(resnet.parameters(), lr=1e-4)
    loss_resnet = train_one_epoch(
        resnet, train_loader, optimizer_resnet, DEVICE, epoch=1
    )

    # Save ResNet checkpoint
    save_checkpoint(
        {"state_dict": resnet.state_dict()},
        is_best=True,
        filename="resnet_checkpoint.pth",
        best_filename="resnet_best.pth",
    )
    assert os.path.exists(
        os.path.join(WORKING_DIR, "resnet_best.pth")
    ), "ResNet checkpoint not saved"

    # B. Train Teacher 2: ConvNeXt
    print("Training ConvNeXt (Teacher 2)...")
    optimizer_convnext = optim.AdamW(convnext.parameters(), lr=1e-4)
    loss_convnext = train_one_epoch(
        convnext, train_loader, optimizer_convnext, DEVICE, epoch=1
    )

    # Save ConvNeXt checkpoint
    save_checkpoint(
        {"state_dict": convnext.state_dict()},
        is_best=True,
        filename="convnext_checkpoint.pth",
        best_filename="convnext_best.pth",
    )
    assert os.path.exists(
        os.path.join(WORKING_DIR, "convnext_best.pth")
    ), "ConvNeXt checkpoint not saved"

    # C. Train Student: MaxViT (Distillation)
    print("Training MaxViT (Student via Distillation)...")
    optimizer_maxvit = optim.AdamW(maxvit.parameters(), lr=1e-4)
    # Teachers must be in eval mode
    resnet.eval()
    convnext.eval()

    loss_distill = train_distill_one_epoch(
        student=maxvit,
        teachers=[resnet, convnext],
        loader=dual_loader,
        optimizer=optimizer_maxvit,
        device=DEVICE,
        epoch=1,
    )

    # Save MaxViT checkpoint
    save_checkpoint(
        {"state_dict": maxvit.state_dict()},
        is_best=True,
        filename="maxvit_checkpoint.pth",
        best_filename="maxvit_best.pth",
    )
    assert os.path.exists(
        os.path.join(WORKING_DIR, "maxvit_best.pth")
    ), "MaxViT checkpoint not saved"

    # 5. Inference Demonstration
    print("\n--- Running Inference Pipeline ---")
    # This function loads the checkpoints we just saved and runs prediction on the test set
    inference_fn(
        resnet_checkpoint="resnet_best.pth",
        convnext_checkpoint="convnext_best.pth",
        maxvit_checkpoint="maxvit_best.pth",
    )

    # 6. Verify Submission
    print("\n--- Verifying Submission ---")
    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError(f"Submission file not found at {SUBMISSION_PATH}")

    df_sub = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")

    # Check columns
    assert (
        "id" in df_sub.columns and "label" in df_sub.columns
    ), "Submission missing required columns"

    # Check ID range (Test set has 2500 images)
    assert len(df_sub) == 2500, f"Expected 2500 predictions, got {len(df_sub)}"
    assert (
        df_sub["id"].dtype == int or df_sub["id"].dtype == np.int64
    ), "ID column should be integer"

    # Check probability range
    assert (
        df_sub["label"].min() >= 0.0 and df_sub["label"].max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
