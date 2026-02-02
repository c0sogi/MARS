import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from scipy.stats import pearsonr
import nltk

# Import provided library modules
from library.config import Config
from library.tokenizer import Tokenizer
from library.dataset import get_dataloaders
from library.model import GFCN
from library.train import train_one_epoch, validate, generate_submission
from library.utils import (
    save_checkpoint,
    load_checkpoint,
    compute_levenshtein,
    AverageMeter,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_failure_analysis(model, loader, tokenizer, device):
    """
    Runs inference on validation set to compute per-sample metrics and correlations.
    """
    model.eval()

    levenshtein_distances = []
    target_lengths = []
    image_widths = []
    image_heights = []

    # Access the underlying dataframe to get file paths for image stats
    val_df = loader.dataset.df

    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)
    print("Collecting predictions and metadata...")

    # We need to align loader batches with dataframe rows.
    # Since shuffle=False for validation loader, order is preserved.
    # We will iterate loader and dataframe index synchronously.

    current_idx = 0

    with torch.no_grad():
        for i, (images, labels, label_lengths) in enumerate(loader):
            images = images.to(device)
            batch_size = images.size(0)

            # Forward pass
            logits = model(images)

            # Decode
            decoded_preds = tokenizer.decode_greedy(logits)

            # Get Ground Truths
            decoded_targets = []
            for j in range(batch_size):
                length = label_lengths[j].item()
                indices = labels[j, :length].cpu().numpy()
                target_str = "".join(
                    [tokenizer.idx_to_char.get(idx, "") for idx in indices]
                )
                decoded_targets.append(target_str)

            # Compute Metrics and Collect Stats
            for j in range(batch_size):
                pred = decoded_preds[j]
                target = decoded_targets[j]

                # Metric: Levenshtein Distance
                dist = nltk.edit_distance(pred, target)
                levenshtein_distances.append(dist)
                target_lengths.append(len(target))

                # Image Stats from Dataframe
                # Path is relative to ./input
                rel_path = val_df.iloc[current_idx + j]["file_path"]
                full_path = os.path.join(Config.INPUT_DIR, rel_path)

                # Read image header only to get dimensions fast
                try:
                    img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        h, w = img.shape[:2]
                        image_widths.append(w)
                        image_heights.append(h)
                    else:
                        image_widths.append(0)
                        image_heights.append(0)
                except:
                    image_widths.append(0)
                    image_heights.append(0)

            current_idx += batch_size

            if i % 20 == 0:
                print(f"Analyzed batch {i}/{len(loader)}")

    # Convert to numpy for correlation
    errors = np.array(levenshtein_distances)
    lens = np.array(target_lengths)
    widths = np.array(image_widths)
    heights = np.array(image_heights)

    # Calculate Correlations
    corr_len, _ = pearsonr(lens, errors)
    corr_width, _ = pearsonr(widths, errors)
    corr_height, _ = pearsonr(heights, errors)

    print("\n[Correlation Analysis]")
    print(f"Correlation (Error vs Target Length): {corr_len:.4f}")
    print(f"Correlation (Error vs Image Width):   {corr_width:.4f}")
    print(f"Correlation (Error vs Image Height):  {corr_height:.4f}")

    print("\n[Interpretation]")
    if abs(corr_len) > 0.3:
        print(
            "-> Strong relationship between molecule complexity (length) and error rate."
        )
    if abs(corr_width) > 0.3:
        print("-> Strong relationship between image width and error rate.")


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for Fast Baseline
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20000  # 20k samples for quick training
    Config.NUM_EPOCHS = 5  # 5 epochs
    Config.BATCH_SIZE = 96  # Higher batch size for A100 efficiency
    Config.NUM_WORKERS = 8

    # Create directories
    Config.create_directories()

    # Set seed
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Initializing Tokenizer and DataLoaders...")
    # Load cached data if available to save time
    tokenizer = Tokenizer(load_cached_data=True)
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, load_cached_data=True
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing GFCN Model...")
    model = GFCN(num_classes=len(tokenizer)).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    # CTC Loss: blank=0 matches Tokenizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    best_metric = float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_metric = validate(model, val_loader, criterion, tokenizer, device)

        # Scheduler Step
        scheduler.step(val_metric)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Time: {elapsed:.0f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Levenshtein: {val_metric:.4f}"
        )

        # Save Best Model
        is_best = val_metric < best_metric
        if is_best:
            best_metric = val_metric
            print(f"New best model found! Score: {best_metric}")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_metric": best_metric,
            },
            is_best,
        )

    # -------------------------------------------------------------------------
    # 5. Final Validation Assessment
    # -------------------------------------------------------------------------
    print("\nLoading best model for final assessment...")
    # Re-initialize and load best weights
    model = GFCN(num_classes=len(tokenizer)).to(device)
    load_checkpoint(model, filename=Config.BEST_MODEL_PATH)

    # Compute Final Validation Metric on the full validation set (subset in debug mode)
    print("Computing final validation metric...")
    _, final_val_metric = validate(model, val_loader, criterion, tokenizer, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    run_failure_analysis(model, val_loader, tokenizer, device)

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    print("\nGenerating submission for test set...")
    generate_submission(model, test_loader, tokenizer, device)

    print("\nRunfile execution complete.")


if __name__ == "__main__":
    main()
