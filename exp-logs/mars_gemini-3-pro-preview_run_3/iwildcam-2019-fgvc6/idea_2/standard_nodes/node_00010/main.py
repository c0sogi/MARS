import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# Import components from the provided library
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_macro_f1,
    save_checkpoint,
    load_checkpoint,
)
from library.dataset import get_dataloaders
from library.model import AnimalClassifier
from library.train import train_one_epoch, validate
from library.predict import generate_submission


def run_failure_analysis(val_df, predictions, targets, ids):
    """
    Analyzes failure modes by correlating prediction error with input image features.
    Samples the validation set to ensure analysis fits within time limits.
    """
    print("\n--- Failure Analysis ---")

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame(
        {"Id": ids, "Target": targets, "Prediction": predictions}
    )

    # Calculate Error (1 = Incorrect, 0 = Correct)
    analysis_df["Error"] = (analysis_df["Target"] != analysis_df["Prediction"]).astype(
        int
    )

    # Merge with metadata to get file paths
    # val_df contains 'Id' and 'file_path'
    analysis_df = analysis_df.merge(val_df[["Id", "file_path"]], on="Id", how="left")

    # Sample data to keep analysis fast (target ~2000 samples)
    SAMPLE_SIZE = 2000
    if len(analysis_df) > SAMPLE_SIZE:
        # Try stratified sampling to capture both errors and correct predictions
        try:
            from sklearn.model_selection import train_test_split

            _, sample_df = train_test_split(
                analysis_df,
                test_size=SAMPLE_SIZE,
                stratify=analysis_df["Error"],
                random_state=Config.SEED,
            )
        except ValueError:
            # Fallback if stratification fails (e.g., too few errors)
            sample_df = analysis_df.sample(n=SAMPLE_SIZE, random_state=Config.SEED)
    else:
        sample_df = analysis_df

    print(f"Computing image features for {len(sample_df)} validation samples...")

    stats = []
    for idx, row in sample_df.iterrows():
        full_path = os.path.join(Config.INPUT_ROOT, row["file_path"])
        try:
            # Read image to get original properties
            img = cv2.imread(full_path)
            if img is not None:
                h, w, c = img.shape
                # Normalize pixel values for stats
                img_float = img.astype(np.float32) / 255.0
                mean_val = np.mean(img_float)
                std_val = np.std(img_float)

                stats.append(
                    {
                        "Id": row["Id"],
                        "Height": h,
                        "Width": w,
                        "AspectRatio": w / h if h > 0 else 0,
                        "PixelMean": mean_val,
                        "PixelStd": std_val,
                    }
                )
        except Exception:
            continue

    stats_df = pd.DataFrame(stats)
    if stats_df.empty:
        print("Warning: Could not compute image features for analysis.")
        return

    # Merge features back with error info
    merged_df = sample_df.merge(stats_df, on="Id")

    # Calculate correlations
    features = ["Height", "Width", "AspectRatio", "PixelMean", "PixelStd"]
    print("\nCorrelation between Error (1=Wrong, 0=Right) and Input Features:")
    for feat in features:
        if feat in merged_df.columns:
            corr = merged_df["Error"].corr(merged_df[feat])
            print(f"{feat}: {corr:.4f}")


def main():
    # 1. Configuration Setup
    # Adjust Config for a fast but effective baseline
    Config.NUM_EPOCHS = 7
    Config.BATCH_SIZE = 64

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model Initialization
    print(f"Initializing Model: {Config.MODEL_NAME}")
    model = AnimalClassifier(pretrained=True)
    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 4. Training Loop
    best_val_f1 = -1.0
    print("Starting Training...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        # Validate
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss:.6f}, Train F1: {train_f1:.6f}")
        print(f"Val Loss: {val_loss:.6f}, Val F1: {val_f1:.6f}")

        # Save Checkpoint
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            print("New best model found. Saving checkpoint.")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_f1": best_val_f1,
            },
            is_best=is_best,
        )

    # 5. Final Evaluation
    print("\nLoading best model for final evaluation...")
    load_checkpoint(Config.MODEL_CHECKPOINT_PATH, model)
    model.eval()

    print("Running full validation inference...")
    all_targets = []
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, targets, ids in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_ids.extend(ids)

    final_metric = calculate_macro_f1(np.array(all_targets), np.array(all_preds))

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    # Load validation metadata to get file paths
    if os.path.exists(Config.VAL_METADATA_PATH):
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)
        run_failure_analysis(val_df, all_preds, all_targets, all_ids)
    else:
        print("Validation metadata not found. Skipping failure analysis.")

    # 7. Submission
    THRESHOLD = 0.9293196996798049

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(
            checkpoint_path=Config.MODEL_CHECKPOINT_PATH,
            output_path=Config.SUBMISSION_PATH,
        )
    else:
        print(
            f"\nMetric ({final_metric}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
