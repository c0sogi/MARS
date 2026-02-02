import os
import cv2
import torch
import numpy as np
import pandas as pd
import warnings
import torch.nn as nn
from library.config import Config
from library.utils import seed_everything, calculate_macro_f1
from library.dataset import get_dataloaders
from library.model import HerbariumResNet
from library.trainer import Trainer

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    print("Setting up environment...")
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    # We run for 1 epoch to ensure completion within the time limit while using the full dataset
    Config.NUM_EPOCHS = 1

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading data...")
    # Load data with caching enabled for speed
    # Disable cache loading to prevent KeyErrors from stale mappings (e.g. from debug runs)
    data = get_dataloaders(load_cached_data=False)

    train_loader = data["train"]
    val_loader = data["val"]
    test_loader = data["test"]
    num_classes = data["num_classes"]
    class_mapping = data["class_mapping"]

    # Create inverse mapping (class_idx -> category_id) for submission
    idx_to_category = {v: k for k, v in class_mapping.items()}

    print(f"Data loaded. Classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("Initializing model...")
    model = HerbariumResNet(num_classes=num_classes)

    # Optimizer & Scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
    )

    # --------------------------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------------------------
    print("Starting training...")
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # --------------------------------------------------------------------------
    # 5. Validation & Metrics
    # --------------------------------------------------------------------------
    print("Performing final validation...")
    model.eval()

    all_preds = []
    all_labels = []

    device = Config.DEVICE

    # We use a manual loop to gather all predictions for analysis
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Use mixed precision for inference
            if device == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
            else:
                outputs = model(images)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate and print the required metric
    macro_f1 = calculate_macro_f1(all_labels, all_preds)
    print(f"Final Validation Metric: {macro_f1}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    print("Performing failure analysis...")

    # Load validation metadata to link predictions with image properties
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment
    if len(val_df) == len(all_preds):
        analysis_df = val_df.copy()
        analysis_df["pred"] = all_preds
        analysis_df["label"] = all_labels
        analysis_df["is_error"] = (analysis_df["pred"] != analysis_df["label"]).astype(
            int
        )

        # Sample a subset for image property extraction to save time
        sample_size = min(2000, len(analysis_df))
        sample_df = analysis_df.sample(n=sample_size, random_state=Config.SEED).copy()

        widths = []
        heights = []

        for _, row in sample_df.iterrows():
            path = os.path.join(Config.INPUT_DIR, row["file_path"])
            try:
                img = cv2.imread(path)
                if img is not None:
                    h, w, _ = img.shape
                    widths.append(w)
                    heights.append(h)
                else:
                    widths.append(np.nan)
                    heights.append(np.nan)
            except:
                widths.append(np.nan)
                heights.append(np.nan)

        sample_df["width"] = widths
        sample_df["height"] = heights
        sample_df["aspect_ratio"] = sample_df["width"] / sample_df["height"]

        # Drop failures in loading
        sample_df.dropna(subset=["width", "height"], inplace=True)

        # Calculate correlations
        if len(sample_df) > 0:
            print(f"Failure Analysis on {len(sample_df)} samples:")
            for feature in ["width", "height", "aspect_ratio"]:
                # Calculate Pearson correlation
                if sample_df[feature].std() > 0 and sample_df["is_error"].std() > 0:
                    corr = np.corrcoef(sample_df["is_error"], sample_df[feature])[0, 1]
                    print(f"Correlation between Error and {feature}: {corr:.4f}")
                else:
                    print(
                        f"Correlation between Error and {feature}: Undefined (zero variance)"
                    )
    else:
        print("Validation set size mismatch. Skipping detailed failure analysis.")

    # --------------------------------------------------------------------------
    # 7. Submission
    # --------------------------------------------------------------------------
    print("Generating submission...")
    model.eval()

    submission_ids = []
    submission_preds = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            if device == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
            else:
                outputs = model(images)

            _, preds = torch.max(outputs, 1)

            submission_preds.extend(preds.cpu().numpy())
            # image_ids is a tensor
            submission_ids.extend(image_ids.numpy())

    # Map class indices back to category_ids
    final_categories = [idx_to_category[p] for p in submission_preds]

    # Create DataFrame
    sub_df = pd.DataFrame({"Id": submission_ids, "Predicted": final_categories})

    # Sort by Id
    sub_df.sort_values("Id", inplace=True)

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
