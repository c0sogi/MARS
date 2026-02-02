import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import library modules
from library.config import Config
from library.utils import seed_everything, optimize_f1_threshold
from library.dataset import get_dataloaders
from library.model import ArtworkModel, ModelEMA
from library.engine import train_one_epoch, valid_one_epoch


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Override Config for fast baseline execution while maintaining performance
    Config.epochs = 5
    Config.batch_size = 64  # Increase batch size for A100 efficiency

    seed_everything(Config.seed)
    Config.setup()

    print(f"Starting run: {Config.run_name}")
    print(f"Device: {Config.device}")
    print(f"Epochs: {Config.epochs}, Batch Size: {Config.batch_size}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing model...")
    model = ArtworkModel(num_classes=Config.num_classes, pretrained=True)
    model.to(Config.device)

    model_ema = None
    if Config.use_ema:
        print("Using Model EMA...")
        model_ema = ModelEMA(model, decay=Config.ema_decay, device=Config.device)

    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.min_lr)

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(1, Config.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, Config.device, epoch, model_ema
        )

        # Validate (using EMA if available)
        eval_model = model_ema.module if model_ema else model
        val_loss, val_preds, val_targets = valid_one_epoch(
            eval_model, val_loader, Config.device
        )

        # Scheduler Step
        scheduler.step()

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(eval_model.state_dict(), Config.model_save_path)
            print(f"Epoch {epoch}: New best model saved with loss {best_val_loss:.4f}")

    # -------------------------------------------------------------------------
    # 5. Final Evaluation
    # -------------------------------------------------------------------------
    print("\nLoading best model for final evaluation...")
    # Re-instantiate to load clean weights
    final_model = ArtworkModel(num_classes=Config.num_classes, pretrained=False)
    final_model.load_state_dict(
        torch.load(Config.model_save_path, map_location=Config.device)
    )
    final_model.to(Config.device)
    final_model.eval()

    # Inference on Validation Set
    val_loss, val_preds, val_targets = valid_one_epoch(
        final_model, val_loader, Config.device
    )

    # Threshold Optimization
    best_thresh, best_f1 = optimize_f1_threshold(val_targets, val_preds)
    print(f"Best Threshold: {best_thresh:.4f}")
    print(f"Final Validation Metric: {best_f1}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude per sample.
    # Error = Mean Absolute Error between predicted probs and binary targets
    all_errors = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Sample a subset for analysis to keep it fast (reading images is I/O bound)
    val_dataset = val_loader.dataset
    val_df = val_dataset.df
    n_samples = len(val_df)

    # Analyze up to 2000 samples
    sample_indices = np.random.choice(
        n_samples, size=min(2000, n_samples), replace=False
    )

    heights = []
    widths = []
    ratios = []
    errors = []

    for idx in sample_indices:
        row = val_df.iloc[idx]
        full_path = row["full_path"]

        try:
            # Read image to get dimensions
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                heights.append(h)
                widths.append(w)
                ratios.append(w / h if h > 0 else 0)
                errors.append(all_errors[idx])
        except Exception:
            continue

    if len(errors) > 0:
        corr_h = np.corrcoef(heights, errors)[0, 1]
        corr_w = np.corrcoef(widths, errors)[0, 1]
        corr_ar = np.corrcoef(ratios, errors)[0, 1]

        print(f"Correlation Error vs Height: {corr_h:.4f}")
        print(f"Correlation Error vs Width: {corr_w:.4f}")
        print(f"Correlation Error vs Aspect Ratio: {corr_ar:.4f}")
    else:
        print("Could not compute correlations (no images processed).")

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    TARGET_METRIC = 0.6106623748931248

    if best_f1 > TARGET_METRIC:
        print("\nMetric threshold met. Generating submission...")

        test_preds = []
        test_ids = []

        final_model.eval()
        with torch.no_grad():
            for images, _, ids in test_loader:
                images = images.to(Config.device)
                outputs = final_model(images)
                probs = torch.sigmoid(outputs)
                test_preds.append(probs.cpu().numpy())
                test_ids.extend(ids)

        test_preds = np.concatenate(test_preds)

        # Binarize predictions using optimal threshold
        binary_preds = (test_preds > best_thresh).astype(int)

        # Format for CSV
        submission_data = []
        for i, img_id in enumerate(test_ids):
            # Get indices of positive predictions
            active_indices = np.where(binary_preds[i] == 1)[0]
            attr_str = " ".join(map(str, active_indices))
            submission_data.append({"id": img_id, "attribute_ids": attr_str})

        sub_df = pd.DataFrame(submission_data)
        sub_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"\nMetric {best_f1} did not meet threshold {TARGET_METRIC}. Submission skipped."
        )


if __name__ == "__main__":
    main()
