import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from library
from library.config import Config
from library.utils import seed_everything, get_score, get_ema_model
from library.dataset import get_loaders
from library.model import AppleDiseaseModel
from library.engine import fit


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_loaders(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # 3. Model Initialization
    model = AppleDiseaseModel(
        model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED
    )
    model.to(device)

    # Initialize EMA
    ema_model = get_ema_model(model)

    # 4. Training Setup
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_MIN_LR
    )

    # 5. Training Loop
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        ema_model=ema_model,
        patience=5,
    )

    # 6. Load Best Model for Validation & Inference
    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    # If EMA was used, the checkpoint contains the EMA weights (handled in engine.py)
    model.load_state_dict(checkpoint)
    model.eval()

    # 7. Validation Assessment
    val_preds = []
    val_targets = []

    # For Failure Analysis
    error_magnitudes = []
    feature_brightness = []
    feature_contrast = []

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Inference
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Loss per sample for failure analysis (mean over classes)
            loss = criterion(logits, targets).mean(dim=1)

            val_preds.append(probs.cpu())
            val_targets.append(targets.cpu())

            error_magnitudes.append(loss.cpu())

            # Compute simple features from normalized tensor for analysis
            # images: (B, 3, H, W)
            # Brightness: Mean of all pixels
            b_feat = images.mean(dim=[1, 2, 3])
            # Contrast: Std of all pixels
            c_feat = images.std(dim=[1, 2, 3])

            feature_brightness.append(b_feat.cpu())
            feature_contrast.append(c_feat.cpu())

    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)
    error_magnitudes = torch.cat(error_magnitudes)
    feature_brightness = torch.cat(feature_brightness)
    feature_contrast = torch.cat(feature_contrast)

    # Calculate Metric
    final_f1 = get_score(val_targets, val_preds, threshold=Config.CONF_THRESHOLD)
    print(f"Final Validation Metric: {final_f1}")

    # 8. Failure Analysis
    print("Failure Analysis:")
    # Correlation between error magnitude and features
    df_analysis = pd.DataFrame(
        {
            "error": error_magnitudes.numpy(),
            "brightness": feature_brightness.numpy(),
            "contrast": feature_contrast.numpy(),
        }
    )

    corr_brightness = df_analysis["error"].corr(df_analysis["brightness"])
    corr_contrast = df_analysis["error"].corr(df_analysis["contrast"])

    print(f"Correlation (Error vs Brightness): {corr_brightness}")
    print(f"Correlation (Error vs Contrast): {corr_contrast}")

    # 9. Submission
    threshold_score = 0.9096474096681636
    if final_f1 > threshold_score:
        test_ids = []
        test_preds = []

        with torch.no_grad():
            for images, _, ids in test_loader:
                images = images.to(device)

                # TTA Strategy: Original, HFlip, VFlip
                # 1. Original
                logits_1 = model(images)
                probs_1 = torch.sigmoid(logits_1)

                # 2. HFlip
                images_h = torch.flip(images, [3])  # Flip width (dim 3)
                logits_2 = model(images_h)
                probs_2 = torch.sigmoid(logits_2)

                # 3. VFlip
                images_v = torch.flip(images, [2])  # Flip height (dim 2)
                logits_3 = model(images_v)
                probs_3 = torch.sigmoid(logits_3)

                # Average
                avg_probs = (probs_1 + probs_2 + probs_3) / 3.0

                test_preds.append(avg_probs.cpu())
                test_ids.extend(ids)

        test_preds = torch.cat(test_preds).numpy()

        # Format predictions
        submission_rows = []
        for i, img_id in enumerate(test_ids):
            probs = test_preds[i]
            # Binarize
            labels_indices = np.where(probs > Config.CONF_THRESHOLD)[0]

            if len(labels_indices) == 0:
                # Fallback: pick max probability if no class meets threshold
                max_idx = np.argmax(probs)
                labels_indices = [max_idx]

            pred_labels = [Config.ID2LABEL[idx] for idx in labels_indices]
            label_str = " ".join(pred_labels)

            submission_rows.append({"image": img_id, "labels": label_str})

        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    main()
