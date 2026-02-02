import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import CFG
from library.utils import seed_everything, get_pos_weights, calculate_metric
from library.data import load_image_dict, BirdDataset, get_transforms
from library.models import get_model
from library.losses import WeightedBCELoss, WeightedDistillationLoss
from library.engine import (
    train_one_epoch,
    valid_one_epoch,
    train_distill_one_epoch,
    predict_with_tta,
)


def main():
    print("Starting Library Demonstration...")

    # 1. Setup and Configuration Overrides for Speed
    seed_everything(42)

    # Modify CFG for a quick demo run
    CFG.batch_size = 4
    CFG.epochs = 1
    CFG.debug = True
    # Ensure working directory exists for caching
    os.makedirs(CFG.working_dir, exist_ok=True)

    print(f"Device: {CFG.device}")

    # 2. Data Loading Demonstration
    print("\n--- Demonstrating Data Loading ---")
    if not os.path.exists(CFG.train_csv):
        raise FileNotFoundError(f"Metadata file not found at {CFG.train_csv}")

    # Load a small subset of data for demonstration
    full_df = pd.read_csv(CFG.train_csv)
    demo_df = (
        full_df.head(12).copy().reset_index(drop=True)
    )  # 12 samples, divisible by batch_size 4
    print(f"Loaded demo subset with {len(demo_df)} samples.")

    # Load images (using a unique cache name to avoid interfering with full training)
    image_dict = load_image_dict(
        demo_df, load_cached_data=False, cache_name="demo_subset"
    )

    # Create Dataset and Loader
    train_ds = BirdDataset(demo_df, image_dict, transforms=get_transforms("train"))
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)

    # Validate Data Shapes
    images, targets = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.shape == (
        CFG.batch_size,
        3,
        CFG.img_height,
        CFG.img_width,
    ), "Incorrect image dimensions"
    assert targets.shape == (
        CFG.batch_size,
        CFG.num_classes,
    ), "Incorrect target dimensions"

    # 3. Model Initialization
    print("\n--- Demonstrating Model Initialization ---")
    # Use pretrained=False to avoid download time/errors during demo
    model = get_model("resnet18", pretrained=False)
    model.to(CFG.device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = images.to(CFG.device)
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (
        CFG.batch_size,
        CFG.num_classes,
    ), "Model output shape mismatch"

    # 4. Loss Function Verification
    print("\n--- Demonstrating Loss Functions ---")
    # Calculate weights based on the demo dataframe
    pos_weights = get_pos_weights(demo_df, device=CFG.device)

    # Standard Weighted BCE
    criterion = WeightedBCELoss(pos_weights=pos_weights, device=CFG.device)
    loss_val = criterion(dummy_output, targets.to(CFG.device))
    print(f"Weighted BCE Loss: {loss_val.item():.4f}")
    assert not torch.isnan(loss_val), "Loss is NaN"

    # 5. Training Engine Demonstration
    print("\n--- Demonstrating Training Loop ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train one epoch
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, CFG.device, epoch=0
    )
    print(f"Train Loss: {train_loss:.4f}")

    # Validate one epoch
    # Create a validation loader (same data for demo purposes, different transforms)
    val_ds = BirdDataset(demo_df, image_dict, transforms=get_transforms("valid"))
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)

    val_loss, val_auc = valid_one_epoch(model, val_loader, criterion, CFG.device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    # 6. Distillation Demonstration
    print("\n--- Demonstrating Knowledge Distillation ---")

    # Create a custom dataset that yields (Image, [Target, TeacherProbs])
    class DistillDataset(BirdDataset):
        def __getitem__(self, idx):
            img, target = super().__getitem__(idx)
            # Create dummy teacher probabilities (Soft targets)
            # In a real scenario, these come from the teacher model's predictions
            teacher_probs = torch.rand_like(target)
            # Pack them: [Target (19), Teacher (19)]
            packed = torch.cat([target, teacher_probs])
            return img, packed

    distill_ds = DistillDataset(demo_df, image_dict, transforms=get_transforms("train"))
    distill_loader = DataLoader(distill_ds, batch_size=CFG.batch_size, shuffle=True)

    distill_criterion = WeightedDistillationLoss(
        pos_weights=pos_weights, device=CFG.device
    )

    # Run distillation training step
    distill_loss = train_distill_one_epoch(
        model, distill_loader, optimizer, distill_criterion, CFG.device, epoch=0
    )
    print(f"Distillation Loss: {distill_loss:.4f}")

    # 7. Inference with TTA
    print("\n--- Demonstrating Inference with TTA ---")
    # Using the validation loader for inference
    preds = predict_with_tta(model, val_loader, CFG.device)

    print(f"Predictions Shape: {preds.shape}")
    print(f"Predictions Range: [{preds.min():.4f}, {preds.max():.4f}]")

    assert preds.shape == (len(demo_df), CFG.num_classes), "Prediction shape mismatch"
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
