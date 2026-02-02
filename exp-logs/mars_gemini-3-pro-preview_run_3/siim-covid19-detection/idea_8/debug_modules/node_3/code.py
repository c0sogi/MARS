import torch
import numpy as np
import os
import sys
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, collate_fn
from library.dataset import SIIMDataset
from library.transforms import get_transforms
from library.model import MultiTaskDeformableDETR
from library.loss import SetCriterion
from library.engine import train_one_epoch, evaluate


def run_pipeline_demo():
    print("=== Starting Multi-Task Deformable DETR Pipeline Demo ===")

    # 1. Configuration & Setup
    # Override Config for speed and debugging purposes
    Config.DEBUG = True
    Config.DEBUG_DATA_SIZE = 10  # Limit to 10 samples for quick execution
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.EPOCHS = 1
    Config.IMG_SIZE = 512  # Reduce image size for faster processing

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")
    print(f"Debug Mode: {Config.DEBUG}, Data Size: {Config.DEBUG_DATA_SIZE}")

    # 2. Data Loading
    print("\n[Data Loading]")
    # Initialize Datasets
    train_dataset = SIIMDataset(split="train", transform=get_transforms("train"))
    val_dataset = SIIMDataset(split="val", transform=get_transforms("val"))

    # Verify Dataset Lengths
    print(f"Train Dataset Length: {len(train_dataset)}")
    print(f"Val Dataset Length: {len(val_dataset)}")
    assert (
        len(train_dataset) == Config.DEBUG_DATA_SIZE
    ), "Train dataset size does not match debug size."

    # Verify Single Sample Structure
    sample_img, sample_target = train_dataset[0]
    print(f"Sample Image Shape: {sample_img.shape}")
    print(f"Sample Target Keys: {list(sample_target.keys())}")

    assert sample_img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected image shape (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {sample_img.shape}"
    assert "boxes" in sample_target, "Target missing 'boxes' key"
    assert "study_label" in sample_target, "Target missing 'study_label' key"

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify Batch Structure
    batch_imgs, batch_targets = next(iter(train_loader))
    print(f"Batch Images Shape: {batch_imgs.shape}")
    assert batch_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Batch image shape incorrect"
    assert len(batch_targets) == Config.BATCH_SIZE, "Batch target length incorrect"

    # 3. Model Initialization
    print("\n[Model Initialization]")
    model = MultiTaskDeformableDETR()
    model.to(device)
    print("Model initialized successfully.")

    # 4. Forward Pass
    print("\n[Forward Pass]")
    batch_imgs = batch_imgs.to(device)

    # Run inference
    outputs = model(batch_imgs)
    print(f"Output Keys: {list(outputs.keys())}")

    # Verify Output Shapes
    # pred_logits: (Batch, Num_Object_Queries, Num_Classes + 1)
    # pred_boxes: (Batch, Num_Object_Queries, 4)
    # pred_study_logits: (Batch, Num_Study_Classes)

    assert outputs["pred_logits"].shape == (
        Config.BATCH_SIZE,
        Config.NUM_OBJECT_QUERIES,
        Config.NUM_OBJECT_CLASSES + 1,
    ), f"pred_logits shape mismatch: {outputs['pred_logits'].shape}"
    assert outputs["pred_boxes"].shape == (
        Config.BATCH_SIZE,
        Config.NUM_OBJECT_QUERIES,
        4,
    ), f"pred_boxes shape mismatch: {outputs['pred_boxes'].shape}"
    assert outputs["pred_study_logits"].shape == (
        Config.BATCH_SIZE,
        Config.NUM_STUDY_CLASSES,
    ), f"pred_study_logits shape mismatch: {outputs['pred_study_logits'].shape}"

    print("Forward pass shapes verified.")

    # 5. Loss Calculation
    print("\n[Loss Calculation]")
    criterion = SetCriterion()
    criterion.to(device)

    # Move targets to device
    batch_targets_device = [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in batch_targets
    ]

    loss_dict = criterion(outputs, batch_targets_device)
    print(f"Loss Components: {list(loss_dict.keys())}")
    print(f"Total Loss: {loss_dict['loss'].item():.4f}")

    assert not torch.isnan(loss_dict["loss"]), "Loss is NaN!"
    assert "loss_ce" in loss_dict, "Missing classification loss"
    assert "loss_bbox" in loss_dict, "Missing bbox loss"
    assert "loss_study" in loss_dict, "Missing study loss"

    # 6. Training Loop Demonstration
    print("\n[Training Loop Demo]")
    # Configure Optimizer (Backbone usually has lower LR)
    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": Config.LR_BACKBONE,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=Config.LR_TRANSFORMER, weight_decay=Config.WEIGHT_DECAY
    )

    # Run one epoch
    train_stats = train_one_epoch(
        model,
        criterion,
        train_loader,
        optimizer,
        device,
        epoch=0,
        max_norm=Config.CLIP_MAX_NORM,
    )
    print(f"Train Stats: {train_stats}")

    # 7. Evaluation Loop Demonstration
    print("\n[Evaluation Loop Demo]")
    eval_stats = evaluate(model, criterion, val_loader, device)
    print(f"Eval Stats: {eval_stats}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_pipeline_demo()
