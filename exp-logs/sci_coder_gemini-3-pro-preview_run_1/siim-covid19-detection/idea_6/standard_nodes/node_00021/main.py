import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data import prepare_train_val_loaders
from library.model import ResNet34FPN
from library.engine import train_model, Engine
from library.inference import generate_submission


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis by correlating model error with input features.
    """
    print("\n==== Running Failure Analysis ====")
    model.eval()

    criterion_cls = nn.CrossEntropyLoss(reduction="none")
    criterion_seg = nn.BCEWithLogitsLoss(reduction="none")

    errors_cls = []
    errors_seg = []

    feature_num_boxes = []
    feature_class_id = []

    # Iterate over validation set to collect errors and features
    with torch.no_grad():
        for images, masks, labels in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            # Forward pass
            cls_logits, seg_logits = model(images)

            # Target preparation
            target_cls = torch.argmax(labels, dim=1)

            # Calculate per-sample losses
            loss_cls = criterion_cls(cls_logits, target_cls)
            # Seg loss is (B, 1, H, W), mean over spatial dims to get per-image scalar
            loss_seg = criterion_seg(seg_logits, masks).mean(dim=(1, 2, 3))

            errors_cls.extend(loss_cls.cpu().numpy())
            errors_seg.extend(loss_seg.cpu().numpy())

            # Extract Features
            # 1. Class ID
            feature_class_id.extend(target_cls.cpu().numpy())

            # 2. Number of boxes (approximate from mask areas or use labels if available)
            # Here we use the sum of the mask as a proxy for "amount of opacity"
            # which correlates with complexity/num_boxes
            mask_sums = masks.sum(dim=(1, 2, 3)).cpu().numpy()
            feature_num_boxes.extend(mask_sums)

    errors_cls = np.array(errors_cls)
    errors_seg = np.array(errors_seg)
    feature_class_id = np.array(feature_class_id)
    feature_num_boxes = np.array(feature_num_boxes)

    # Calculate Correlations
    # 1. Correlation between Classification Error and Class ID
    corr_cls_id, _ = pearsonr(errors_cls, feature_class_id)
    print(f"Correlation (Cls Error vs Class ID): {corr_cls_id:.4f}")

    # 2. Correlation between Segmentation Error and Opacity Area (proxy for num boxes/size)
    corr_seg_area, _ = pearsonr(errors_seg, feature_num_boxes)
    print(f"Correlation (Seg Error vs Opacity Area): {corr_seg_area:.4f}")

    # 3. Correlation between Classification Error and Opacity Area
    corr_cls_area, _ = pearsonr(errors_cls, feature_num_boxes)
    print(f"Correlation (Cls Error vs Opacity Area): {corr_cls_area:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    # 5 Epochs is sufficient to verify the pipeline and get a decent score on A100
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 32  # Increase batch size for A100 efficiency

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Backbone: {Config.BACKBONE}")

    # 2. Data Loading
    train_loader, val_loader = prepare_train_val_loaders(debug=False)

    # 3. Model Initialization
    model = ResNet34FPN()
    model.to(Config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 4. Training
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        Config.EPOCHS,
        Config.DEVICE,
    )

    # 5. Final Validation & Metrics
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(
        torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
    )

    engine = Engine(model, Config.DEVICE)
    val_loss, val_map = engine.validate(val_loader)

    print(f"Final Validation Metric: {val_map}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, Config.DEVICE)

    # 7. Submission
    THRESHOLD = 0.4915615987761658
    if val_map > THRESHOLD:
        print(
            f"\nValidation mAP ({val_map}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(Config.DEVICE)
    else:
        print(
            f"\nValidation mAP ({val_map}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
