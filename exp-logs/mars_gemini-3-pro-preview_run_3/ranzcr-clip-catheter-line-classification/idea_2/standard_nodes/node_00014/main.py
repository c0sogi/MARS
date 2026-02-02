import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2

# Import library modules
from library.config import Config
from library.data import get_dataloaders
from library.model import MultiTaskModel
from library.engine import fit, generate_submission
from library.utils import seed_everything, get_score


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Adjust configuration for fast baseline execution
    # Using 10 epochs (Cite solution_lesson_node_00010) to allow Dice Loss to converge
    Config.setup()
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, img_size=Config.IMG_SIZE, load_cached_data=True
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    model = MultiTaskModel(
        backbone_name=Config.BACKBONE, num_classes=Config.NUM_CLASSES
    )
    model = model.to(Config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Ensure scheduler matches the updated epoch count
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
    )

    # -------------------------------------------------------------------------
    # 5. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    # Load best model for evaluation
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        )

    model.eval()

    val_preds = []
    val_targets = []

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(Config.DEVICE)

            # Standard forward pass
            cls_logits, _ = model(images)
            probs_1 = torch.sigmoid(cls_logits)

            # TTA: Horizontal Flip
            images_flip = torch.flip(images, dims=[3])
            cls_logits_flip, _ = model(images_flip)
            probs_2 = torch.sigmoid(cls_logits_flip)

            # Average predictions
            probs = (probs_1 + probs_2) / 2.0

            val_preds.append(probs.cpu().numpy())
            val_targets.append(targets.numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    final_metric = get_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("Performing failure analysis...")

    # Calculate Mean Absolute Error per sample
    errors = np.mean(np.abs(val_targets - val_preds), axis=1)

    # Collect metadata (Width, Height, Aspect Ratio)
    # Iterate over the validation dataframe used by the loader
    val_df = val_loader.dataset.df
    widths = []
    heights = []
    aspect_ratios = []

    # Read image dimensions
    for _, row in val_df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # Read image to get dimensions
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 0)
            else:
                widths.append(np.nan)
                heights.append(np.nan)
                aspect_ratios.append(np.nan)
        except Exception:
            widths.append(np.nan)
            heights.append(np.nan)
            aspect_ratios.append(np.nan)

    # Create analysis dataframe
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
        }
    )

    # Drop failures
    df_analysis = df_analysis.dropna()

    print("Correlation between Error Magnitude and Input Features:")
    for feature in ["width", "height", "aspect_ratio"]:
        if df_analysis[feature].std() > 0:
            # Calculate correlation
            corr = np.corrcoef(df_analysis["error"], df_analysis[feature])[0, 1]
            print(f"{feature}: {corr:.4f}")
        else:
            print(f"{feature}: NaN (No variance)")

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.9502318142936872

    if final_metric > THRESHOLD:
        print("Validation metric exceeds threshold. Generating submission...")
        generate_submission(model, test_loader, Config.DEVICE)
    else:
        print(
            f"Validation metric {final_metric} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
