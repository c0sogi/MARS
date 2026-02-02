import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr
import importlib
import library.config

# Force reload of config to handle persistent environment caching (Cite debug_lesson_1)
importlib.reload(library.config)

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders, get_inference_gallery_loader
from library.model import WhaleEfficientNetArcFace
from library.loss import get_loss
from library.engine import fit, extract_features, calculate_map5, predict


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    # Load training and validation loaders
    # load_cached_data=True allows reusing the LabelEncoder if it exists
    # We set it to False here to ensure we don't use a stale cache (e.g. from a debug run)
    train_loader, val_loader, label_encoder = get_dataloaders(load_cached_data=False)

    # We also need a clean gallery loader (Train set, no augs) for validation metrics
    gallery_loader = get_inference_gallery_loader(load_cached_data=False)

    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    model = WhaleEfficientNetArcFace(num_classes=num_classes)
    model.to(device)

    # -------------------------------------------------------------------------
    # 4. Training Setup
    # -------------------------------------------------------------------------
    # Optimizer: AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    # Loss: ArcFace (CrossEntropy on angular margins)
    criterion = get_loss()

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    # engine.fit handles the loop, validation per epoch, and saving the best model
    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        label_encoder=label_encoder,
        epochs=Config.NUM_EPOCHS,
    )

    # -------------------------------------------------------------------------
    # 6. Final Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("Calculating final validation metrics...")

    # Extract features using the best model
    val_feats, val_labels = extract_features(model, val_loader, device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)

    # Calculate MAP@5 using the full pipeline (QE + Re-ranking)
    final_metric = calculate_map5(
        val_feats, val_labels, gallery_feats, gallery_labels, use_pipeline=True
    )

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 7. Failure Analysis
    # -------------------------------------------------------------------------
    print("Performing failure analysis...")

    # Compute similarity matrix for analysis
    # feats are L2 normalized, so dot product is cosine similarity
    sims = np.dot(val_feats, gallery_feats.T)

    errors = []

    # Calculate per-instance error (1.0 - Reciprocal Rank)
    for i in range(len(val_labels)):
        q_label = val_labels[i]
        # Sort indices by similarity descending
        sorted_indices = np.argsort(sims[i])[::-1]

        top_ids = []
        seen_ids = set()
        rank = -1

        # Find rank of the correct label among unique IDs
        for idx in sorted_indices:
            g_label = gallery_labels[idx]
            if g_label not in seen_ids:
                top_ids.append(g_label)
                seen_ids.add(g_label)

            if g_label == q_label:
                rank = len(top_ids)  # 1-based rank
                break

            # Optimization: If we haven't found it in top 100, assume it's far down
            if len(top_ids) > 100:
                break

        # MAP@5 logic: score is 1/rank if rank <= 5, else 0
        # Error metric: We can use 1 - (1/rank) if found, else 1.0
        if rank != -1 and rank <= 5:
            score = 1.0 / rank
        else:
            score = 0.0

        errors.append(1.0 - score)

    errors = np.array(errors)

    # Load validation metadata to get image properties
    df_val = pd.read_csv(Config.VAL_CSV)
    df_val = df_val[df_val["Id"] != "new_whale"].reset_index(drop=True)

    # Collect image stats (Width, Height, Aspect Ratio)
    widths = []
    heights = []
    aspect_ratios = []

    for idx, row in df_val.iterrows():
        fpath = os.path.join(Config.INPUT_ROOT, row["file_path"])
        img = cv2.imread(fpath)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
        else:
            # Fallback (should not happen given metadata validation)
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    # Calculate Correlations
    if len(errors) == len(widths):
        corr_w, _ = pearsonr(errors, widths)
        corr_h, _ = pearsonr(errors, heights)
        corr_ar, _ = pearsonr(errors, aspect_ratios)

        print(f"Correlation (Error vs Width): {corr_w:.4f}")
        print(f"Correlation (Error vs Height): {corr_h:.4f}")
        print(f"Correlation (Error vs Aspect Ratio): {corr_ar:.4f}")
    else:
        print(
            "Mismatch in validation set size and metadata. Skipping correlation analysis."
        )

    # -------------------------------------------------------------------------
    # 8. Submission Generation
    # -------------------------------------------------------------------------
    # Threshold defined in task
    THRESHOLD_METRIC = 0.8543859649122806

    if final_metric > THRESHOLD_METRIC:
        print(f"Metric {final_metric} > {THRESHOLD_METRIC}. Generating submission...")
        # engine.predict handles the full inference pipeline (QE + Re-ranking)
        predict(model, gallery_loader, label_encoder, device)
    else:
        print(
            f"Metric {final_metric} <= {THRESHOLD_METRIC}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
