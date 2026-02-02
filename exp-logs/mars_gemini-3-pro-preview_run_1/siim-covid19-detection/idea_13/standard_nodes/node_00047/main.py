import sys
import os
import time
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import ast
from torch.utils.data import DataLoader

# Ensure the current directory is in the path so library imports work
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.dataset import SIIMDataset, get_transforms
from library.model import ResNet18_UNet
from library.loss import HybridLoss
from library.training import train_one_epoch, valid_one_epoch
from library.inference import predict_and_submit


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Removed Epoch override to allow full training (Cite solution_lesson_node_00002)

    print(f"Starting execution on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Using load_cached_data=True as requested
    train_dataset = SIIMDataset(
        "train", load_cached_data=True, transform=get_transforms("train")
    )
    val_dataset = SIIMDataset(
        "val", load_cached_data=True, transform=get_transforms("val")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = ResNet18_UNet(num_classes=Config.NUM_CLASSES, pretrained=True).to(device)
    criterion = HybridLoss().to(device)

    # Using conservative learning rate and weight decay as per idea
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 4. Training Loop
    best_map = 0.0
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        # Validate
        val_loss, val_map = valid_one_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_map:.6f}"
        )

        # Checkpoint
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 5. Final Metric Reporting
    # Strictly formatted output as required
    print(f"Final Validation Metric: {best_map}")

    # 6. Failure Analysis
    print("\n==== Failure Analysis ====")
    # Load best model for analysis
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    results = []

    # Calculate per-sample error magnitudes
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)
            study_ids = batch["study_id"]

            cls_logits, seg_logits = model(images)

            # Classification Error: 1.0 - Probability of the true class
            probs = torch.softmax(cls_logits, dim=1)
            true_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            cls_errors = 1.0 - true_class_probs.cpu().numpy()

            # Segmentation Error: 1.0 - Dice Score
            pred_masks = (torch.sigmoid(seg_logits) > 0.5).float()
            intersection = (pred_masks * masks).sum(dim=(1, 2, 3))
            cardinality = pred_masks.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection) / (cardinality + 1e-6)
            seg_errors = 1.0 - dice.cpu().numpy()

            for i in range(len(study_ids)):
                results.append(
                    {
                        "study_id": study_ids[i],
                        "cls_error": cls_errors[i],
                        "seg_error": seg_errors[i],
                    }
                )

    # Merge with metadata to get input features
    df_results = pd.DataFrame(results)
    df_meta = val_dataset.df.copy()

    # Feature 1: Number of Bounding Boxes
    def get_num_boxes(x):
        try:
            return len(ast.literal_eval(x))
        except:
            return 0

    df_meta["num_boxes"] = df_meta["boxes"].apply(get_num_boxes)

    # Feature 2: Class Label Index
    label_cols = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]
    df_meta["label_idx"] = df_meta[label_cols].values.argmax(axis=1)

    # Merge results with metadata
    df_analysis = pd.merge(
        df_results,
        df_meta[["study_id", "num_boxes", "label_idx"]],
        on="study_id",
        how="left",
    )
    # Handle potential duplicates if multiple images per study (though typical in val is 1)
    df_analysis = df_analysis.drop_duplicates(subset=["study_id"])

    # Compute Correlations
    cols_to_corr = ["cls_error", "seg_error", "label_idx", "num_boxes"]
    corr_matrix = df_analysis[cols_to_corr].corr()

    print("Correlation between Error Magnitude and Input Features:")
    print(corr_matrix)

    # 7. Submission Generation
    threshold = 0.49944536565378
    if best_map > threshold:
        print(
            f"\nValidation mAP ({best_map:.6f}) > Threshold ({threshold}). Generating submission..."
        )
        predict_and_submit(load_cached_data=True)
    else:
        print(
            f"\nValidation mAP ({best_map:.6f}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
