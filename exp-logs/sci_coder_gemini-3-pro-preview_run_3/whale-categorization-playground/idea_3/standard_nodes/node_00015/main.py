import os
import sys
import cv2
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import CFG
from library.utils import seed_everything, apk
from library.dataset import WhaleDataset, get_transforms
from library.models import WhaleEfficientNet
from library.losses import CurricularFaceLoss
from library.engine import fit, predict_and_submit


def analyze_failures(model, dataloader, device, id_map):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input features (confidence, aspect ratio).
    """
    model.eval()
    idx_to_id = {v: k for k, v in id_map.items()}

    errors = []
    confidences = []
    aspect_ratios = []

    # Access head for logits to calculate confidence
    if hasattr(model, "module"):
        head_layer = model.module.head
    else:
        head_layer = model.head

    # Get metadata dataframe for file paths to read image dimensions
    df_val = dataloader.dataset.df
    current_idx = 0

    print("Analyzing validation failures...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            bs = images.size(0)

            # Get embeddings and logits
            embeddings = model(images)
            logits = head_layer(embeddings)

            # Get top 5 predictions for metric calculation
            top_scores, topk_indices = torch.topk(logits, k=5, dim=1)
            topk_indices = topk_indices.cpu().numpy()
            top_scores = top_scores.cpu().numpy()

            labels_np = labels.numpy()

            for i in range(bs):
                true_idx = labels_np[i]

                # Decode Ground Truth
                # If true_idx is -1 (unknown class in val), map to 'new_whale'
                if true_idx in idx_to_id:
                    true_label = idx_to_id[true_idx]
                else:
                    true_label = "new_whale"

                # Decode Predictions
                preds = [idx_to_id.get(idx, "new_whale") for idx in topk_indices[i]]

                # Calculate Error (1 - Precision)
                # apk returns precision (0 to 1), so error is 1 - apk
                precision = apk(true_label, preds, k=5)
                errors.append(1.0 - precision)

                # Confidence (Max Logit value as proxy)
                confidences.append(top_scores[i][0])

                # Aspect Ratio
                # We read the original image to get accurate AR
                try:
                    row = df_val.iloc[current_idx]
                    fpath = os.path.join(CFG.input_dir, row["file_path"])
                    # Use cv2.IMREAD_UNCHANGED for speed, we only need shape
                    img = cv2.imread(fpath)
                    if img is not None:
                        h, w = img.shape[:2]
                        aspect_ratios.append(w / h)
                    else:
                        aspect_ratios.append(0.0)
                except Exception:
                    aspect_ratios.append(0.0)

                current_idx += 1

    # Compute Correlations
    if len(errors) > 0:
        df_res = pd.DataFrame(
            {"error": errors, "confidence": confidences, "aspect_ratio": aspect_ratios}
        )

        # Filter out invalid aspect ratios if any
        df_res = df_res[df_res["aspect_ratio"] > 0]

        if not df_res.empty:
            corr_conf = df_res["error"].corr(df_res["confidence"])
            corr_ar = df_res["error"].corr(df_res["aspect_ratio"])

            print(f"Correlation (Error vs Confidence): {corr_conf:.4f}")
            print(f"Correlation (Error vs Aspect Ratio): {corr_ar:.4f}")
        else:
            print("Insufficient data for correlation analysis.")


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    seed_everything(CFG.seed)
    device = CFG.device

    # Fast Baseline Settings: Limit epochs to ensure completion within 2 hours
    epochs = 12

    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    # Train: Exclude 'new_whale' to learn clean ID embeddings
    train_dataset = WhaleDataset(
        csv_file=CFG.train_csv,
        mode="train",
        transform=get_transforms("train"),
        exclude_new_whale=True,
    )

    # Val: Include 'new_whale' for realistic evaluation
    # Reuse id_map from training to ensure consistent class indices
    val_dataset = WhaleDataset(
        csv_file=CFG.val_csv,
        mode="val",
        transform=get_transforms("val"),
        id_map=train_dataset.get_id_map(),
        exclude_new_whale=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    num_classes = len(train_dataset.get_id_map())
    print(f"Training on {len(train_dataset)} images with {num_classes} classes.")
    print(f"Validating on {len(val_dataset)} images.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    model = WhaleEfficientNet(num_classes=num_classes)
    model.to(device)

    # -------------------------------------------------------------------------
    # 4. Training Setup
    # -------------------------------------------------------------------------
    criterion = CurricularFaceLoss(s=CFG.s, m=CFG.m).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=CFG.min_lr
    )

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("Starting training...")
    best_score = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs=epochs,
        patience=5,
        save_path=CFG.model_path,
    )

    # Required Output Format
    print(f"Final Validation Metric: {best_score}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    analyze_failures(model, val_loader, device, train_dataset.get_id_map())

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    # Threshold from requirements
    SUBMISSION_THRESHOLD = 0.8543859649122806

    if best_score > SUBMISSION_THRESHOLD:
        print(
            f"Validation score ({best_score}) exceeds threshold. Generating submission..."
        )

        # Load Test Data
        test_dataset = WhaleDataset(
            csv_file=CFG.test_csv, mode="test", transform=get_transforms("test")
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        # Gallery Loader (Training data, no shuffle)
        # We need to extract features for all training data to use as the gallery
        gallery_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        predict_and_submit(
            model,
            gallery_loader,
            test_loader,
            device,
            train_dataset.get_id_map(),
            CFG.submission_path,
        )
        print(f"Submission generated at {CFG.submission_path}")
    else:
        print(
            f"Validation score ({best_score}) did not meet threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
