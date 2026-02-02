import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import cv2
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_kappa_score, ModelEMA
from library.dataset import create_dataloaders
from library.model import RetinaModel
from library.engine import train_one_epoch, validate, inference_tta


def run():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Override Config for fast baseline execution
    Config.epochs = 10
    Config.batch_size = 8

    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    # Set seeds
    seed_everything(Config.seed)

    print(f"Initializing training on device: {Config.device}")
    print(
        f"Model: {Config.model_name}, Image Size: {Config.image_size}, Epochs: {Config.epochs}"
    )

    # ==========================================
    # 2. Data Loading
    # ==========================================
    train_loader, val_loader, test_loader = create_dataloaders(load_cached_data=True)

    # ==========================================
    # 3. Model & Optimizer Initialization
    # ==========================================
    model = RetinaModel().to(Config.device)

    # Initialize EMA
    ema = (
        ModelEMA(model, decay=Config.ema_decay, device=Config.device)
        if Config.use_ema
        else None
    )

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    # Loss & Scaler
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=Config.use_amp)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_kappa = -float("inf")

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, ema, Config.device
        )

        # Validate (use EMA model)
        val_model = ema.module if ema else model
        val_loss, val_kappa = validate(val_model, val_loader, criterion, Config.device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Kappa: {val_kappa:.4f}"
        )

        # Save Best Model
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(val_model.state_dict(), Config.best_model_path)

    # ==========================================
    # 5. Final Evaluation & Failure Analysis
    # ==========================================
    print("\nStarting Final Evaluation & Failure Analysis...")

    # Load Best Model
    best_model = RetinaModel()
    best_model.load_state_dict(
        torch.load(Config.best_model_path, map_location=Config.device)
    )
    best_model.to(Config.device)
    best_model.eval()

    # Generate predictions on validation set
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(Config.device)

            # Forward pass
            outputs = best_model(images)
            probs = torch.sigmoid(outputs)

            # Decode: Sum and round
            preds = probs.sum(dim=1).round()

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.numpy())

    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)

    # Compute Final Metric
    final_kappa = compute_kappa_score(val_labels, val_preds)
    print(f"Final Validation Metric: {final_kappa}")

    # Failure Analysis: Correlation with Input Features
    print("\nPerforming Failure Analysis...")
    val_df = pd.read_csv(Config.val_csv)

    # Calculate absolute errors
    errors = np.abs(val_labels - val_preds)

    # Extract meta-features from images
    widths, heights, ratios, intensities = [], [], [], []

    # Note: We iterate through val_df to read original image stats
    # We assume val_df order matches val_loader order (which is shuffle=False)
    if len(val_df) != len(errors):
        # Handle potential subsetting in Config.debug or Config.val_subset_size
        # If lengths mismatch, we truncate to the smaller length for analysis
        min_len = min(len(val_df), len(errors))
        val_df = val_df.iloc[:min_len]
        errors = errors[:min_len]

    for idx, row in val_df.iterrows():
        full_path = os.path.join(Config.input_dir, row["file_path"])

        # Read image to get stats
        try:
            img = cv2.imread(full_path)
            if img is None:
                # Dummy values if image read fails
                widths.append(0)
                heights.append(0)
                ratios.append(0)
                intensities.append(0)
                continue

            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            ratios.append(w / h)
            intensities.append(img.mean())
        except Exception:
            widths.append(0)
            heights.append(0)
            ratios.append(0)
            intensities.append(0)

    # Compute Correlations
    meta_features = {
        "width": widths,
        "height": heights,
        "aspect_ratio": ratios,
        "mean_intensity": intensities,
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, data in meta_features.items():
        if len(data) > 1:
            corr, _ = pearsonr(data, errors)
            print(f"{name}: {corr:.4f}")
        else:
            print(f"{name}: N/A (Insufficient data)")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    threshold = 0.9234606183435836

    if final_kappa > threshold:
        print(
            f"\nMetric ({final_kappa}) > Threshold ({threshold}). Generating submission..."
        )

        # Inference with TTA
        test_preds = inference_tta(best_model, test_loader, Config.device)

        # Create Submission DataFrame
        test_df = pd.read_csv(Config.test_csv)

        # Handle subsetting if applicable
        if len(test_preds) != len(test_df):
            test_df = test_df.iloc[: len(test_preds)]

        submission = pd.DataFrame(
            {"id_code": test_df["id_code"], "diagnosis": test_preds}
        )

        # Save to ./submission/submission.csv
        sub_path = "./submission/submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric ({final_kappa}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    run()
