import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import ast

# Import from provided library
from library.config import Config
from library.dataset import SIIMDataset
from library.model import DropBlockResNet34UNet
from library.engine import train_one_epoch, evaluate, predict_test
from library.utils import seed_everything


def failure_analysis(model, loader, device, metadata_df):
    """
    Performs failure analysis by correlating classification error with input features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    errors = []
    feature_num_boxes = []
    feature_is_negative = []

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    # Create a lookup for metadata to speed up
    # Ensure unique index
    meta_lookup = metadata_df.drop_duplicates(subset=["image_id"]).set_index("image_id")

    with torch.no_grad():
        for data in loader:
            images = data["image"].to(device)
            labels = data["labels"].to(device)
            image_ids = data["image_id"]

            # Calculate Classification Error (Cross Entropy)
            cls_targets = torch.argmax(labels, dim=1)
            cls_logits, _ = model(images)
            loss = criterion(cls_logits, cls_targets)

            errors.extend(loss.cpu().numpy())

            for img_id in image_ids:
                if img_id in meta_lookup.index:
                    row = meta_lookup.loc[img_id]

                    # Feature 1: Number of boxes (Complexity)
                    boxes_str = row.get("boxes", np.nan)
                    if pd.isna(boxes_str):
                        n_b = 0
                    else:
                        try:
                            n_b = len(ast.literal_eval(boxes_str))
                        except:
                            n_b = 0
                    feature_num_boxes.append(n_b)

                    # Feature 2: Is Negative Label
                    is_neg = row.get("Negative for Pneumonia", 0)
                    feature_is_negative.append(is_neg)
                else:
                    feature_num_boxes.append(0)
                    feature_is_negative.append(0)

    # Calculate Correlations
    if len(errors) > 1:
        corr_boxes = np.corrcoef(errors, feature_num_boxes)[0, 1]
        corr_neg = np.corrcoef(errors, feature_is_negative)[0, 1]

        print(f"Correlation between Error (CE Loss) and Num Boxes: {corr_boxes:.4f}")
        print(
            f"Correlation between Error (CE Loss) and Is Negative Label: {corr_neg:.4f}"
        )
    else:
        print("Not enough data for correlation analysis.")


def run_pipeline():
    # 1. Configuration
    Config.setup()

    # Override for fast baseline execution
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 32
    # Update LR based on scaling rule
    Config.LEARNING_RATE = Config.REF_LR * (Config.BATCH_SIZE / Config.REF_BATCH_SIZE)

    # Ensure reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading Datasets...")
    # Using full dataset to maximize performance within the 2h limit (A100 is fast)
    train_dataset = SIIMDataset("train", load_cached_data=True)
    val_dataset = SIIMDataset("val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Setup
    print("Initializing Model...")
    model = DropBlockResNet34UNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 4. Training Loop
    best_map = 0.0

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # Step Scheduler
        scheduler.step()

        # Validate
        val_map = evaluate(model, val_loader, device, val_dataset.df)

        # Save Best
        if val_map > best_map:
            print(f"New best mAP: {val_map:.6f}")
            best_map = val_map
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 5. Final Evaluation & Failure Analysis
    print("Loading best model for final evaluation...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No model checkpoint found. Using current model state.")

    # Calculate final metric on full validation set
    final_metric = evaluate(model, val_loader, device, val_dataset.df)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    failure_analysis(model, val_loader, device, val_dataset.df)

    # 6. Submission
    THRESHOLD = 0.49944536565378
    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = SIIMDataset("test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        predict_test(model, test_loader, device, test_dataset.df)
    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
