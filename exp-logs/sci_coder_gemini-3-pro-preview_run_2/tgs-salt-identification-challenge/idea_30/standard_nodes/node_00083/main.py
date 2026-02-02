import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library components
from library.utils import set_seed
from library.dataset import SaltDataset, ORIG_SIZE, TARGET_SIZE
from library.model import ResNet34WideLinkNet
from library.losses import BCEWithLovaszLoss
from library.engine import fit, validate_one_epoch
from library.inference import (
    optimize_threshold,
    generate_submission,
    calculate_map_vectorized,
)

# --- Configuration ---
SEED = 42
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING_DIR = "./working"
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_PATH = "./submission/submission.csv"


def main():
    # 1. Setup
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Running on device: {DEVICE}")

    # 2. Data Loading
    # We use the pre-generated metadata splits (train.csv / val.csv)
    print("Initializing datasets...")
    train_dataset = SaltDataset(mode="train")
    val_dataset = SaltDataset(mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing ResNet34WideLinkNet model...")
    model = ResNet34WideLinkNet(pretrained=True)
    model.to(DEVICE)

    # 4. Training Configuration
    criterion = BCEWithLovaszLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5. Training Loop
    print("Starting training...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        epochs=EPOCHS,
        patience=10,
        save_dir=CHECKPOINT_DIR,
    )

    # 6. Evaluation & Threshold Optimization
    print("\n--- Evaluation ---")
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    print(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    # Optimize threshold
    best_threshold = optimize_threshold(model, val_loader, DEVICE)

    # 7. Final Metric Calculation & Failure Analysis
    print("\n--- Failure Analysis ---")

    # Run validation inference to get raw logits for analysis
    dummy_criterion = torch.nn.BCEWithLogitsLoss()
    _, val_logits, val_targets, val_ids = validate_one_epoch(
        model, val_loader, dummy_criterion, DEVICE, epoch=999
    )

    # Squeeze channels if necessary
    if val_logits.ndim == 4:
        val_logits = val_logits.squeeze(1)
    if val_targets.ndim == 4:
        val_targets = val_targets.squeeze(1)

    # Crop to original size (101x101) for accurate metric calculation
    pad_t = (TARGET_SIZE - ORIG_SIZE) // 2
    pad_l = (TARGET_SIZE - ORIG_SIZE) // 2

    logits_cropped = val_logits[:, pad_t : pad_t + ORIG_SIZE, pad_l : pad_l + ORIG_SIZE]
    targets_cropped = val_targets[
        :, pad_t : pad_t + ORIG_SIZE, pad_l : pad_l + ORIG_SIZE
    ]

    # Convert to probabilities and binary predictions
    probs = 1.0 / (1.0 + np.exp(-logits_cropped))
    preds_bool = probs > best_threshold
    targets_bool = targets_cropped > 0.5

    # Calculate and print the required metric
    final_metric = calculate_map_vectorized(preds_bool, targets_bool)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Perform Failure Analysis (Correlations)
    # Calculate per-image scores
    N = preds_bool.shape[0]
    p_flat = preds_bool.reshape(N, -1)
    t_flat = targets_bool.reshape(N, -1)

    pred_empty = p_flat.sum(axis=1) == 0
    gt_empty = t_flat.sum(axis=1) == 0

    intersection = (p_flat & t_flat).sum(axis=1)
    union = (p_flat | t_flat).sum(axis=1)

    iou = np.zeros(N, dtype=np.float32)
    valid_union = union > 0
    iou[valid_union] = intersection[valid_union] / union[valid_union]

    thresholds = np.arange(0.5, 0.95 + 1e-6, 0.05)
    matches = iou[:, None] > thresholds[None, :]
    score_iou = matches.mean(axis=1)

    scores = np.zeros(N, dtype=np.float32)
    scores[pred_empty & gt_empty] = 1.0
    both_non_empty = (~pred_empty) & (~gt_empty)
    scores[both_non_empty] = score_iou[both_non_empty]

    # Load metadata to correlate errors
    val_df = pd.read_csv("./metadata/val.csv")

    # Create analysis dataframe
    # Note: val_ids from loader matches the order of predictions
    analysis_df = pd.DataFrame({"id": val_ids, "score": scores, "error": 1.0 - scores})

    # Merge with metadata to get 'z' and 'salt_coverage'
    analysis_df = analysis_df.merge(val_df, on="id", how="left")

    # Calculate correlations
    corr_depth = analysis_df["error"].corr(analysis_df["z"])
    corr_coverage = analysis_df["error"].corr(analysis_df["salt_coverage"])

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_coverage:.4f}")

    # 8. Submission Generation
    if final_metric > 0.7985:
        print("\nMetric threshold passed. Generating submission...")
        test_dataset = SaltDataset(mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        generate_submission(
            model=model,
            test_loader=test_loader,
            threshold=best_threshold,
            device=DEVICE,
            save_path=SUBMISSION_PATH,
        )
    else:
        print(f"\nFinal metric {final_metric:.4f} is below 0.7985. Submission skipped.")


if __name__ == "__main__":
    main()
