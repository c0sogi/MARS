import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_dataloaders
from library.model import get_model, get_optimizer, get_scheduler, get_loss_fn
from library.train import train_one_epoch, validate, predict
from library.inference import create_submission

# =============================================================================
# Configuration Overrides for Fast Baseline
# =============================================================================
# Reduce epochs to ensure the script completes quickly within the time limit.
# ConvNeXt-Tiny pre-trained on ImageNet converges very quickly on this dataset.
Config.EPOCHS = 5
Config.SCHEDULER_T_MAX = Config.EPOCHS
Config.WORKING_DIR = "./working/idea_2"
os.makedirs(Config.WORKING_DIR, exist_ok=True)

logger = get_logger("runfile")


# =============================================================================
# Failure Analysis Function
# =============================================================================
def analyze_failure(model, val_loader, val_csv_path, device):
    """
    Performs failure analysis on the validation set by correlating prediction
    errors with image metadata features.
    """
    logger.info("Starting Failure Analysis...")

    # 1. Get Predictions on Validation Set
    model.eval()
    preds = []
    targets = []

    # val_loader is assumed to be deterministic (shuffle=False)
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            # Probability of Class 1 (Dog)
            dog_probs = probs[:, 1].cpu().numpy()

            preds.extend(dog_probs)
            targets.extend(labels.numpy())

    # Load validation metadata
    val_df = pd.read_csv(val_csv_path)

    # Ensure lengths match
    if len(val_df) != len(preds):
        logger.warning(
            f"Mismatch in validation set size: DF {len(val_df)} vs Preds {len(preds)}"
        )
        # Truncate to minimum length to avoid crash, though this shouldn't happen
        min_len = min(len(val_df), len(preds))
        val_df = val_df.iloc[:min_len]
        preds = preds[:min_len]
        targets = targets[:min_len]

    val_df["prob_dog"] = preds
    val_df["target"] = targets

    # Calculate Error Magnitude
    # Error = |Target - Predicted_Prob|
    val_df["error"] = np.abs(val_df["target"] - val_df["prob_dog"])

    # 2. Extract Metadata Features (Width, Height, Aspect Ratio, File Size)
    widths = []
    heights = []
    aspect_ratios = []
    file_sizes = []

    # Iterate through files to get current metadata
    for idx, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["filepath"])

        if os.path.exists(full_path):
            # File Size
            fsize = os.path.getsize(full_path)
            file_sizes.append(fsize)

            # Dimensions
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 0)
            else:
                widths.append(0)
                heights.append(0)
                aspect_ratios.append(0)
        else:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    val_df["width"] = widths
    val_df["height"] = heights
    val_df["aspect_ratio"] = aspect_ratios
    val_df["file_size"] = file_sizes

    # 3. Compute and Print Correlations
    features = ["width", "height", "aspect_ratio", "file_size"]
    print("\n--- Failure Analysis: Error Correlation ---")
    for feat in features:
        if val_df[feat].std() > 0:
            corr, _ = pearsonr(val_df[feat], val_df["error"])
            print(f"Correlation between Error and {feat}: {corr:.6f}")
        else:
            print(f"Correlation between Error and {feat}: NaN (Constant feature)")
    print("-------------------------------------------\n")


# =============================================================================
# Main Execution Flow
# =============================================================================
def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model Initialization
    model = get_model()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    criterion = get_loss_fn()

    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    logger.info(f"Starting training for {Config.EPOCHS} epochs on {Config.DEVICE}...")

    # 4. Training Loop
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )

        # Validate
        val_loss = validate(model, val_loader, Config.DEVICE)

        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Log Loss: {val_loss:.5f}"
        )

        # Scheduler Step
        scheduler.step()

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Metric Output (Required Format)
    print(f"Final Validation Metric: {best_val_loss}")

    # 6. Failure Analysis
    # Load best model weights
    logger.info("Loading best model for analysis and inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

    analyze_failure(model, val_loader, Config.VAL_CSV, Config.DEVICE)

    # 7. Conditional Submission
    THRESHOLD = 0.014050961788691994

    if best_val_loss < THRESHOLD:
        logger.info(
            f"Validation metric {best_val_loss} < {THRESHOLD}. Generating submission..."
        )
        ids, probs = predict(model, test_loader, Config.DEVICE, use_tta=Config.USE_TTA)
        create_submission(ids, probs, Config.SUBMISSION_PATH)
    else:
        logger.info(
            f"Validation metric {best_val_loss} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
