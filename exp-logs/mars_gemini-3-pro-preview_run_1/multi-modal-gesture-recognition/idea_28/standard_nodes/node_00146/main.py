import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, compute_levenshtein, median_filter, rle_decode
from library.data_loader import get_dataloaders
from library.model import BAMPNet
from library.train import train_epoch, validate, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Limit epochs for fast baseline execution within time limits
    Config.NUM_EPOCHS = 10

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    train_loader, val_loader, test_loader = get_dataloaders()

    # Load Ground Truth and Metadata for Validation (for Failure Analysis)
    val_df = pd.read_csv(Config.VAL_CSV)
    gt_map = {}
    val_meta = {}
    for _, row in val_df.iterrows():
        lbls = row["labels"]
        if pd.isna(lbls) or lbls == "":
            gt_map[row["sample_id"]] = []
            num_gestures = 0
        else:
            gt_map[row["sample_id"]] = [int(float(x)) for x in str(lbls).split(",")]
            num_gestures = len(gt_map[row["sample_id"]])

        val_meta[row["sample_id"]] = {
            "num_frames": row["num_frames"],
            "num_gestures": num_gestures,
        }

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = BAMPNet().to(device)

    # Loss Setup
    # Aggressive down-weighting for background class (Cite Lesson 00095)
    class_weights = torch.ones(Config.NUM_CLASSES + 1).to(device)
    class_weights[Config.BACKGROUND_CLASS_ID] = 0.35

    criterion_class = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING
    )
    criterion_boundary = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_ler = float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_epoch(
            model, train_loader, optimizer, criterion_class, criterion_boundary, device
        )

        # Validate
        val_ler = validate(model, val_loader, gt_map, device)

        # Scheduler Step
        scheduler.step()

        # Checkpoint
        if val_ler < best_ler:
            best_ler = val_ler
            torch.save(model.state_dict(), best_model_path)

    # ==========================================
    # 5. Final Evaluation & Failure Analysis
    # ==========================================
    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()

    # Re-run validation to collect per-sample errors for analysis
    sample_errors = []
    sample_ids_list = []

    total_dist = 0
    total_gestures = 0

    with torch.no_grad():
        for batch in val_loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"]
            ids = batch["ids"]

            outputs = model(skeleton, audio, lengths, mask)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=2)
            preds = torch.argmax(probs, dim=2).cpu().numpy()

            for i, sample_id in enumerate(ids):
                valid_len = lengths[i]
                sample_pred = preds[i, :valid_len]

                # Post-processing
                sample_pred = median_filter(sample_pred, window_size=5)
                pred_seq = rle_decode(
                    sample_pred,
                    background_class_id=Config.BACKGROUND_CLASS_ID,
                    min_duration=5,
                )

                gt_seq = gt_map.get(sample_id, [])
                dist = compute_levenshtein(pred_seq, gt_seq)

                n_gestures = len(gt_seq)
                total_dist += dist
                total_gestures += n_gestures

                # Per-sample metric for correlation
                if n_gestures > 0:
                    sample_metric = dist / n_gestures
                else:
                    # If GT is empty but model predicted something, error is the insertion count (dist)
                    sample_metric = float(dist)

                sample_errors.append(sample_metric)
                sample_ids_list.append(sample_id)

    # Compute Global Metric
    final_metric = total_dist / total_gestures if total_gestures > 0 else 1.0

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    frames_list = [val_meta[sid]["num_frames"] for sid in sample_ids_list]
    ngestures_list = [val_meta[sid]["num_gestures"] for sid in sample_ids_list]

    df_analysis = pd.DataFrame(
        {
            "error": sample_errors,
            "num_frames": frames_list,
            "num_gestures": ngestures_list,
        }
    )

    # Calculate correlations (handle NaNs if variance is 0)
    corr_frames = df_analysis["error"].corr(df_analysis["num_frames"])
    corr_gestures = df_analysis["error"].corr(df_analysis["num_gestures"])

    if pd.isna(corr_frames):
        corr_frames = 0.0
    if pd.isna(corr_gestures):
        corr_gestures = 0.0

    print("Failure Analysis - Correlations with Error:")
    print(f"  Sequence Length (Frames): {corr_frames:.4f}")
    print(f"  Number of Gestures: {corr_gestures:.4f}")

    # ==========================================
    # 6. Submission
    # ==========================================
    threshold = 0.05697278911564626
    if final_metric < threshold:
        generate_submission(model, test_loader, device, Config.SUBMISSION_FILE)


if __name__ == "__main__":
    main()
