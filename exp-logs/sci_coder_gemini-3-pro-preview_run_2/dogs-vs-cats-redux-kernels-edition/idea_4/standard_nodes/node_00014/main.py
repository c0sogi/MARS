import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import create_dataloaders
from library.model import DogCatClassifier, ModelEMA
from library.engine import train_one_epoch, validate, predict_tta

# Suppress warnings
warnings.filterwarnings("ignore")


def get_validation_predictions(model, loader, device):
    """
    Runs inference on the validation set to get probabilities and targets.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    return np.concatenate(all_preds).flatten(), np.concatenate(all_targets).flatten()


def perform_failure_analysis(val_df, preds, targets):
    """
    Analyzes the correlation between prediction error and image metadata.
    """
    print("Performing failure analysis...")

    # Calculate error magnitude (absolute difference)
    errors = np.abs(targets - preds)

    # We need to gather image dimensions.
    # Since this is a fast baseline, we read images on the fly.
    widths = []
    heights = []
    aspect_ratios = []

    input_dir = Config.INPUT_DIR

    # Iterate through the dataframe matching the prediction order
    for _, row in val_df.iterrows():
        filepath = os.path.join(input_dir, row["filepath"])
        img = cv2.imread(filepath)

        if img is not None:
            h, w, _ = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
        else:
            # Handle missing/corrupt images
            widths.append(np.nan)
            heights.append(np.nan)
            aspect_ratios.append(np.nan)

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
        }
    )

    # Drop rows with failed image reads
    analysis_df = analysis_df.dropna()

    print("Correlation between Error Magnitude and Input Features:")
    for feature in ["width", "height", "aspect_ratio"]:
        # Calculate Pearson correlation
        corr = analysis_df["error"].corr(analysis_df[feature])
        print(f"  {feature}: {corr:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()
    logger = get_logger()
    device = torch.device(Config.DEVICE)
    logger.info(f"Device: {device}")

    # 2. Data Loading
    # Using debug=False to ensure we train on the full dataset for best performance
    train_loader, val_loader, test_loader = create_dataloaders(debug=False)

    # 3. Model Initialization
    model = DogCatClassifier(pretrained=True).to(device)

    # Initialize EMA
    ema = None
    if Config.USE_EMA:
        ema = ModelEMA(model, decay=Config.EMA_DECAY)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pth")

    logger.info("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, ema_model=ema
        )

        # Validate (use EMA model if available)
        val_model_curr = ema.get_model() if ema else model
        val_loss, val_acc = validate(val_model_curr, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val Acc: {val_acc:.6f}"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(val_model_curr.state_dict(), best_model_path)
            logger.info(f"New best model saved with loss {best_val_loss:.6f}")

    # 6. Final Evaluation
    # Print metric with full precision as required
    print(f"Final Validation Metric: {best_val_loss}")

    # Load best model for analysis and inference
    best_model = DogCatClassifier(pretrained=False)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.to(device)

    # 7. Failure Analysis
    # Get predictions on validation set
    val_preds, val_targets = get_validation_predictions(best_model, val_loader, device)

    # Perform analysis
    perform_failure_analysis(val_loader.dataset.df, val_preds, val_targets)

    # 8. Submission
    THRESHOLD = 0.018199009307556684

    if best_val_loss < THRESHOLD:
        logger.info(
            f"Validation metric ({best_val_loss}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions using TTA
        submission_df = predict_tta(best_model, test_loader, device)

        # Sort and save
        submission_df = submission_df.sort_values("id")
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        logger.info(
            f"Validation metric ({best_val_loss}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
