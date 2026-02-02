import os
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_box_from_mask
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import StochasticDepthResNet34UNet
from library.engine import train_one_epoch, validate, inference


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override epochs for fast baseline execution within time limit
    Config.EPOCHS = 5

    print(f"Initializing run on {device} for {Config.EPOCHS} epochs...")

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = StochasticDepthResNet34UNet()
    model = model.to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    best_map = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # Validate
        val_loss, val_map = validate(model, val_loader, device)

        # Checkpoint
        if val_map > best_map:
            print(
                f"New Best mAP: {val_map:.6f} (Previous: {best_map:.6f}). Saving model..."
            )
            best_map = val_map
            torch.save(model.state_dict(), best_model_path)

    # Required Output Format
    print(f"Final Validation Metric: {best_map}")

    # 6. Failure Analysis
    print("\n==== Failure Analysis ====")
    # Load best model for analysis
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()

    errors = []
    num_boxes_list = []
    class_indices = []

    # Disable gradients for analysis
    with torch.no_grad():
        for images, labels, masks in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Forward pass
            cls_logits, _ = model(images)
            cls_probs = torch.softmax(cls_logits, dim=1)

            # Calculate Error: 1.0 - probability assigned to the correct class
            # labels shape: (B,) containing indices 0-3
            # gather retrieves the prob at the correct index
            true_class_probs = cls_probs.gather(1, labels.view(-1, 1)).squeeze()
            batch_errors = 1.0 - true_class_probs.cpu().numpy()

            errors.extend(batch_errors)
            class_indices.extend(labels.cpu().numpy())

            # Extract features: Number of GT boxes
            masks_np = masks.cpu().numpy()
            for i in range(len(images)):
                # Use utility to count boxes in GT mask
                boxes, _ = get_box_from_mask(masks_np[i, 0], threshold=0.5)
                num_boxes_list.append(len(boxes))

    # Compute Correlations
    if len(errors) > 1:
        # Correlation with Number of Opacities (Complexity)
        corr_boxes, _ = pearsonr(errors, num_boxes_list)
        print(f"Correlation (Error vs Num_GT_Boxes): {corr_boxes:.4f}")

        # Correlation with Class Index (Systematic Class Bias)
        corr_class, _ = pearsonr(errors, class_indices)
        print(f"Correlation (Error vs Class_Index):  {corr_class:.4f}")
    else:
        print("Insufficient validation data for correlation analysis.")

    # 7. Submission
    SUBMISSION_THRESHOLD = 0.49944536565378

    if best_map > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation metric {best_map:.6f} > {SUBMISSION_THRESHOLD}. Generating submission..."
        )

        # Load Test Loader
        test_loader = get_test_dataloader(load_cached_data=True)

        # Run Inference (Includes TTA and Gating)
        inference(model, test_loader, device)
    else:
        print(
            f"\nValidation metric {best_map:.6f} <= {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
