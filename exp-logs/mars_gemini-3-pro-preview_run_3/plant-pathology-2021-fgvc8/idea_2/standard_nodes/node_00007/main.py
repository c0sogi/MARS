import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import (
    load_and_process_data,
    get_transforms,
    AppleDataset,
    MixupCutmix,
)
from library.model import AppleClassifier
from library.loss import AsymmetricLoss
from library.engine import fit, validate, inference_tta


def get_validation_predictions(model, loader, device):
    """
    Runs inference on validation set to get predictions and targets for failure analysis.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def perform_failure_analysis(df, preds, targets):
    """
    Calculates error magnitude and correlates it with image metadata.
    """
    print("\n--- Failure Analysis ---")

    # Calculate Error Magnitude (Mean Absolute Error per sample)
    # shape: (N_samples, N_classes) -> (N_samples,)
    error_magnitude = np.mean(np.abs(preds - targets), axis=1)

    # Extract metadata features for the validation set
    # We do this on the fly to avoid modifying the cached metadata files
    widths = []
    heights = []
    brightnesses = []

    print("Extracting metadata features for validation images...")
    for idx, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_ROOT, row["file_path"])
        try:
            img = cv2.imread(file_path)
            if img is not None:
                h, w, _ = img.shape
                # Simple brightness estimation
                b = np.mean(img) / 255.0

                widths.append(w)
                heights.append(h)
                brightnesses.append(b)
            else:
                # Fallback if image load fails (shouldn't happen based on checks)
                widths.append(0)
                heights.append(0)
                brightnesses.append(0)
        except Exception:
            widths.append(0)
            heights.append(0)
            brightnesses.append(0)

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {
            "error": error_magnitude,
            "width": widths,
            "height": heights,
            "brightness": brightnesses,
        }
    )

    # Calculate correlations
    features = ["width", "height", "brightness"]
    print("\nCorrelation between Error Magnitude and Image Features:")
    for feat in features:
        if analysis_df[feat].std() > 0:
            corr, _ = pearsonr(analysis_df["error"], analysis_df[feat])
            print(f"  - {feat.capitalize()}: {corr:.4f}")
        else:
            print(f"  - {feat.capitalize()}: N/A (No variance)")


def main():
    # 1. Setup
    cfg = Config()
    seed_everything(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading metadata...")
    # Load cached data or process from scratch
    train_df = load_and_process_data(
        cfg.TRAIN_METADATA, "train_cache", load_cached_data=True
    )
    val_df = load_and_process_data(cfg.VAL_METADATA, "val_cache", load_cached_data=True)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Create Datasets
    train_ds = AppleDataset(train_df, transform=get_transforms("train", cfg))
    val_ds = AppleDataset(val_df, transform=get_transforms("val", cfg))

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    # 3. Model Initialization
    print(f"Initializing model: {cfg.MODEL_NAME}")
    model = AppleClassifier(
        model_name=cfg.MODEL_NAME,
        num_classes=cfg.NUM_CLASSES,
        pretrained=cfg.PRETRAINED,
        dropout_rate=cfg.DROPOUT_RATE,
    )
    model.to(device)

    # 4. Training Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.T_MAX, eta_min=cfg.MIN_LR
    )

    if cfg.USE_ASL:
        loss_fn = AsymmetricLoss(
            gamma_neg=cfg.ASL_GAMMA_NEG, gamma_pos=cfg.ASL_GAMMA_POS, clip=cfg.ASL_CLIP
        )
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    mixup_fn = None
    if cfg.USE_MIXUP or (cfg.CUTMIX_ALPHA > 0 and cfg.MIX_PROB > 0):
        mixup_fn = MixupCutmix(cfg)

    # 5. Training Loop
    print("Starting training...")
    best_model_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")

    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=cfg.EPOCHS,
        mixup_fn=mixup_fn,
        loss_fn=loss_fn,
        patience=5,  # Early stopping patience
        save_path=best_model_path,
    )

    # 6. Final Evaluation
    print("Evaluating best model...")
    # Ensure best weights are loaded (fit returns the model with best weights)
    val_loss, val_f1 = validate(model, val_loader, loss_fn, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_f1}")

    # 7. Failure Analysis
    print("Generating predictions for failure analysis...")
    val_preds, val_targets = get_validation_predictions(model, val_loader, device)
    perform_failure_analysis(val_df, val_preds, val_targets)

    # 8. Submission
    SUBMISSION_THRESHOLD = 0.8854796331311056

    if val_f1 > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation F1 ({val_f1:.4f}) > Threshold ({SUBMISSION_THRESHOLD:.4f}). Generating submission..."
        )

        # Load Test Data
        test_df = load_and_process_data(
            cfg.TEST_METADATA, "test_cache", load_cached_data=True
        )
        test_ds = AppleDataset(
            test_df, transform=get_transforms("test", cfg), return_id=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
        )

        # Generate Submission using TTA
        inference_tta(model, test_loader, device, cfg.SUBMISSION_PATH)
        print(f"Submission saved to {cfg.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation F1 ({val_f1:.4f}) did not meet threshold ({SUBMISSION_THRESHOLD:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
