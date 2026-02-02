import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Import provided library modules
from library.config import Config
from library.dataset import (
    get_train_val_loaders,
    get_test_loader,
    load_and_process_metadata,
)
from library.model import AppleDiseaseModel
from library.engine import train_model, validate, predict_tta
from library.utils import seed_everything, ModelEMA


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    # Enable cudnn benchmark for speed since input sizes are fixed
    torch.backends.cudnn.benchmark = True
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Using full dataset to ensure high performance
    train_loader, val_loader = get_train_val_loaders(Config, load_cached_data=True)

    # 3. Model Initialization
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = AppleDiseaseModel(Config).to(device)

    # 4. Optimizer & Scheduler
    # Using AdamW as specified in Idea
    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        eps=Config.EPS,
        betas=Config.BETAS,
    )

    # Scheduler: Linear Warmup -> Cosine Annealing
    # T_max for Cosine is total epochs - warmup epochs
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS - Config.WARMUP_EPOCHS, eta_min=Config.MIN_LR
    )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=Config.WARMUP_EPOCHS
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[Config.WARMUP_EPOCHS],
    )

    # 5. EMA
    model_ema = None
    if Config.USE_EMA:
        print(f"Initializing Model EMA (decay={Config.EMA_DECAY})")
        model_ema = ModelEMA(model, decay=Config.EMA_DECAY, device=device)

    # 6. Training
    print(f"Starting training for {Config.EPOCHS} epochs...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        checkpoint_dir=Config.CHECKPOINT_DIR,
        use_ema=Config.USE_EMA,
        model_ema=model_ema,
    )

    # 7. Final Validation & Failure Analysis
    print("\n=== Final Validation & Failure Analysis ===")

    # Load Best Model
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    # Ensure model is in eval mode
    model.eval()

    # Compute Final Metric
    val_loss, val_f1 = validate(model, val_loader, device, threshold=Config.THRESHOLD)
    print(f"Final Validation Metric: {val_f1}")

    # Failure Analysis
    print("Performing failure analysis...")

    # Get per-sample predictions and targets
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            # Standard forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # Calculate Mean Absolute Error per sample (averaged across classes)
    # Shape: (N_samples,)
    per_sample_error = np.abs(all_probs - all_targets).mean(axis=1)

    # Load Validation Metadata to access file paths
    val_df = load_and_process_metadata(Config.VAL_METADATA_PATH, Config.VAL_CACHE_PATH)

    # Calculate image stats and correlate
    # We iterate through the dataframe. Order is preserved as val_loader (shuffle=False).
    meta_stats = []

    # Pre-check if lengths match
    if len(val_df) != len(per_sample_error):
        print(
            f"Warning: Mismatch in validation set size. DF: {len(val_df)}, Preds: {len(per_sample_error)}"
        )

    for idx, row in val_df.iterrows():
        if idx >= len(per_sample_error):
            break

        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image for stats
        try:
            img = cv2.imread(file_path)
            if img is not None:
                h, w, c = img.shape
                brightness = np.mean(img)
                meta_stats.append(
                    {
                        "width": w,
                        "height": h,
                        "aspect_ratio": w / h if h > 0 else 0,
                        "brightness": brightness,
                        "error": per_sample_error[idx],
                    }
                )
        except Exception:
            pass

    stats_df = pd.DataFrame(meta_stats)

    if not stats_df.empty:
        print("\nCorrelation between Error Magnitude and Input Features:")
        features = ["width", "height", "aspect_ratio", "brightness"]
        for feat in features:
            if feat in stats_df.columns:
                corr = stats_df[feat].corr(stats_df["error"])
                print(f"{feat}: {corr:.4f}")
    else:
        print("Could not compute correlations (empty stats).")

    # 8. Submission
    TARGET_METRIC = 0.9096474096681636

    if val_f1 > TARGET_METRIC:
        print(
            f"\nValidation F1 ({val_f1:.6f}) exceeds threshold ({TARGET_METRIC:.6f}). Generating submission..."
        )

        # Load Test Data
        test_loader = get_test_loader(Config, load_cached_data=True)

        # Generate Predictions with TTA
        submission_df = predict_tta(
            model, test_loader, device, threshold=Config.THRESHOLD
        )

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation F1 ({val_f1:.6f}) did not exceed threshold ({TARGET_METRIC:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
