import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import load_dataset_dataframe, AppleDataset, get_transforms
from library.model import AppleMaxViT
from library.engine import fit, predict


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure output directory exists
    os.makedirs("./submission", exist_ok=True)

    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    print("Loading metadata...")
    # Load metadata with caching enabled
    df_train = load_dataset_dataframe(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_CACHE_PATH
    )
    df_val = load_dataset_dataframe(Config.VAL_METADATA_PATH, Config.VAL_CACHE_PATH)
    df_test = load_dataset_dataframe(Config.TEST_METADATA_PATH, Config.TEST_CACHE_PATH)

    # Initialize Datasets
    # Train uses augmentation, Val/Test use simple resizing/normalization
    train_dataset = AppleDataset(df_train, transforms=get_transforms("train"))
    val_dataset = AppleDataset(df_val, transforms=get_transforms("valid"))
    test_dataset = AppleDataset(df_test, transforms=get_transforms("valid"))

    # Initialize DataLoaders
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("Initializing model...")
    model = AppleMaxViT(pretrained=True)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    print("Starting training...")
    # fit() handles the training loop, validation monitoring, and saving best model
    fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs=Config.EPOCHS,
    )

    # -------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("Loading best model for analysis...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Generate predictions on validation set for analysis
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                logits = model(images)
                probs = torch.sigmoid(logits)

            val_probs.append(probs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

    val_probs = np.concatenate(val_probs)
    val_targets = np.concatenate(val_targets)

    # Calculate and Print Final Metric
    final_f1 = get_score(val_targets, val_probs, threshold=Config.THRESHOLD)
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    print("Performing failure analysis...")
    # Calculate Mean Absolute Error per image
    errors = np.abs(val_targets - val_probs).mean(axis=1)

    # Extract image metadata (dimensions) for correlation analysis
    widths = []
    heights = []
    aspect_ratios = []

    for _, row in df_val.iterrows():
        # Construct full path
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            with Image.open(full_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h != 0 else 0)
        except Exception:
            # Fallback for any missing/corrupt images in analysis
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    # Compute Correlations
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
        }
    )

    # Filter out invalid entries if any
    analysis_df = analysis_df[analysis_df["width"] > 0]

    if not analysis_df.empty:
        corr_w = analysis_df["error"].corr(analysis_df["width"])
        corr_h = analysis_df["error"].corr(analysis_df["height"])
        corr_ar = analysis_df["error"].corr(analysis_df["aspect_ratio"])

        print(f"Correlation between Error and Width: {corr_w}")
        print(f"Correlation between Error and Height: {corr_h}")
        print(f"Correlation between Error and Aspect Ratio: {corr_ar}")
    else:
        print("Could not compute correlations (empty data).")

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    THRESHOLD_SCORE = 0.920968

    if final_f1 > THRESHOLD_SCORE:
        print(
            f"Validation metric ({final_f1}) meets threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        # Generate submission using the engine's predict function
        # This saves to Config.SUBMISSION_PATH (./working/idea_5/submission.csv)
        predict(model, test_loader, device)

        # Move submission to the required location
        src_path = Config.SUBMISSION_PATH
        dst_path = "./submission/submission.csv"

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            print(f"Submission saved to {dst_path}")
        else:
            print(f"Error: Submission file not found at {src_path}")
    else:
        print(
            f"Validation metric ({final_f1}) does not meet threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
