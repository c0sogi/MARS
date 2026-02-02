import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from PIL import Image

from library import config
from library import utils
from library import data
from library import model as lib_model
from library import engine
from library import inference


def main():
    # 1. Setup Environment
    utils.set_seed(config.SEED)
    device = config.DEVICE

    # 2. Prepare Data
    # We use the full dataset to maximize performance given the A100 capability.
    # Training 18k images on A100 with ConvNeXt Tiny is sufficiently fast.
    train_loader, val_loader, test_loader = data.get_dataloaders(
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
    )

    # 3. Initialize Model
    model = lib_model.create_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
    )
    model = model.to(device)

    # 4. Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=config.ETA_MIN
    )

    # 5. Train Model
    # engine.train_model handles the training loop, validation, and checkpointing
    best_loss = engine.train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs=config.EPOCHS,
        mixup_alpha=config.MIXUP_ALPHA,
    )

    # 6. Report Validation Metric
    # The task requires printing the full precision metric
    print(f"Final Validation Metric: {best_loss}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")

    # Load the best model state for analysis
    checkpoint = utils.load_checkpoint("model_best.pth")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Get predictions on validation set
    # We need to correlate errors with metadata, so we need row-wise correspondence.
    # val_loader is created with shuffle=False, so it matches the order of val_df.
    val_df = pd.read_csv(config.VAL_META)

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # Forward pass
            outputs = model(images).view(-1)
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.numpy())

    val_df["prob"] = all_probs
    val_df["target"] = all_targets
    val_df["error"] = np.abs(val_df["target"] - val_df["prob"])

    # Extract metadata features (width, height, aspect ratio)
    widths = []
    heights = []
    aspect_ratios = []

    # We iterate through the dataframe to open images and get dimensions
    for _, row in val_df.iterrows():
        img_path = os.path.join(config.INPUT_DIR, row["filepath"])
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 0)
        except Exception:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    val_df["width"] = widths
    val_df["height"] = heights
    val_df["aspect_ratio"] = aspect_ratios

    # Calculate correlations
    corr_width = val_df["error"].corr(val_df["width"])
    corr_height = val_df["error"].corr(val_df["height"])
    corr_ar = val_df["error"].corr(val_df["aspect_ratio"])

    print("Correlation between Error Magnitude and Input Features:")
    print(f"  Width: {corr_width}")
    print(f"  Height: {corr_height}")
    print(f"  Aspect Ratio: {corr_ar}")

    # 8. Submission Generation
    # Condition: Validation metric must be lower than the threshold
    THRESHOLD = 0.018199009307556684

    if best_loss < THRESHOLD:
        print(
            f"Validation metric {best_loss} meets threshold {THRESHOLD}. Generating submission..."
        )
        inference.run_inference(
            checkpoint_name="model_best.pth",
            output_path=config.SUBMISSION_PATH,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            device=device,
        )
    else:
        print(
            f"Validation metric {best_loss} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
