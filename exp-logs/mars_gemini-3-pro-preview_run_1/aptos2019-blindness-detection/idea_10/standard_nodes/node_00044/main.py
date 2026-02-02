import sys
import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

# Import provided library components
from library.config import Config, seed_everything
from library.dataset import get_loaders
from library.model import MultiScaleConvNeXt
from library.utils import quadratic_weighted_kappa, ModelEMA
from library.train import train_fn, eval_fn, inference_fn


def perform_failure_analysis(val_loader, y_true, y_pred_continuous):
    """
    Analyzes the correlation between prediction error and input image metadata.
    """
    print("\n=== Failure Analysis ===")

    # Calculate absolute error based on rounded predictions (classification error)
    y_pred_rounded = np.round(y_pred_continuous)
    errors = np.abs(y_true - y_pred_rounded)

    # Retrieve metadata from the validation dataframe
    val_df = val_loader.dataset.df
    input_dir = Config.input_dir

    widths = []
    heights = []
    file_sizes = []

    # We need to read image files to get original dimensions and size
    print("Extracting metadata from validation images for analysis...")
    for _, row in val_df.iterrows():
        # Construct full path
        file_path = os.path.join(input_dir, row["file_path"])
        try:
            # File size
            f_size = os.path.getsize(file_path)
            file_sizes.append(f_size)

            # Dimensions
            img = cv2.imread(file_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(0)
                heights.append(0)
        except Exception:
            widths.append(0)
            heights.append(0)
            file_sizes.append(0)

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {"error": errors, "width": widths, "height": heights, "file_size": file_sizes}
    )

    # Calculate correlations
    print("Correlation between Model Error Magnitude and Input Features:")
    for feature in ["width", "height", "file_size"]:
        if analysis_df[feature].std() > 0:
            corr, _ = pearsonr(analysis_df["error"], analysis_df[feature])
            print(f"{feature}: {corr:.4f}")
        else:
            print(f"{feature}: NaN (No variance)")


def main():
    # 1. Configuration and Setup
    # Extend training to 15 epochs as per lesson on stochastic depth
    # Cite solution_lesson_node_00043
    Config.epochs = 15

    # Increase batch size to 24 for stability on A100
    # Cite solution_lesson_node_00007
    Config.batch_size = 24

    seed_everything(Config.seed)
    device = Config.device
    print(f"Running on device: {device}")
    print(f"Batch Size set to: {Config.batch_size}")

    # 2. Data Loading
    # Load cached data to speed up initialization
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    model = MultiScaleConvNeXt(pretrained=True).to(device)

    # EMA Setup
    ema_helper = None
    if Config.use_ema:
        ema_helper = ModelEMA(model, decay=Config.ema_decay)
        print("Model EMA initialized.")

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_score = -1.0
    best_model_path = os.path.join(Config.working_dir, "best_model.pth")

    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(Config.epochs):
        start_time = time.time()

        # Train Step
        train_loss = train_fn(
            model, ema_helper, train_loader, criterion, optimizer, device, epoch
        )

        # Validation Step (Use EMA model)
        val_model = ema_helper.get_model() if ema_helper else model
        val_score, _, _ = eval_fn(val_model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.epochs} - Loss: {train_loss:.4f} - QWK: {val_score:.4f} - Time: {elapsed:.0f}s"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(val_model.state_dict(), best_model_path)

    # 5. Final Evaluation and Analysis
    print(f"Final Validation Metric: {best_score}")

    # Load best model for analysis and inference
    print("Loading best model for analysis...")
    final_model = MultiScaleConvNeXt(pretrained=False)
    final_model.load_state_dict(torch.load(best_model_path, map_location=device))
    final_model.to(device)
    final_model.eval()

    # Get predictions on validation set
    val_score, val_preds, val_true = eval_fn(final_model, val_loader, device)

    # Perform Failure Analysis
    perform_failure_analysis(val_loader, val_true, val_preds)

    # 6. Submission Generation
    TARGET_THRESHOLD = 0.9277

    if best_score > TARGET_THRESHOLD:
        print(
            f"Validation score ({best_score}) exceeds threshold ({TARGET_THRESHOLD}). Generating submission..."
        )

        # Inference on Test Set
        test_scores = inference_fn(final_model, test_loader, device)

        # Process Predictions (Round and Clip)
        test_labels = np.round(test_scores).astype(int)
        test_labels = np.clip(test_labels, 0, 4)

        # Create Submission DataFrame
        df_test = test_loader.dataset.df
        submission = pd.DataFrame(
            {"id_code": df_test["id_code"], "diagnosis": test_labels}
        )

        # Save
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"Validation score ({best_score}) did not exceed threshold ({TARGET_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
