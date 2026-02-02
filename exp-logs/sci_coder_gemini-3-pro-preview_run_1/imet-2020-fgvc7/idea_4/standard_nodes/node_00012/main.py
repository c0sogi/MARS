import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, ModelEMA, save_checkpoint
from library.dataset import get_dataloaders
from library.model import ArtworkModel
from library.engine import train_one_epoch, validate, predict


def run():
    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    start_time = time.time()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # Ensure submission directory exists as per prompt requirement
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)
    print("Data loaded successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = ArtworkModel(pretrained=Config.PRETRAINED)
    model.to(device)

    ema_model = None
    if Config.USE_EMA:
        print(f"Initializing Model EMA (decay={Config.EMA_DECAY})")
        ema_model = ModelEMA(model, decay=Config.EMA_DECAY)

    # -------------------------------------------------------------------------
    # 4. Optimizer & Scheduler
    # -------------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR)

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    best_f1 = 0.0
    best_epoch = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()

        # Train Step
        train_loss = train_one_epoch(
            model, ema_model, optimizer, train_loader, device, epoch
        )

        # Validation Step
        # Use EMA model for validation if available, as it's expected to be more robust
        val_model_ref = ema_model.module if ema_model else model
        val_loss, val_f1, _, _ = validate(val_model_ref, val_loader, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val F1 (th=0.5): {val_f1:.5f}"
        )

        # Save Best Checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": val_model_ref.state_dict(),
                    "best_score": best_f1,
                },
                is_best=True,
            )
            print(f"--> New Best Score: {best_f1:.5f}")

        epoch_duration = time.time() - epoch_start
        total_elapsed = time.time() - start_time
        print(
            f"Epoch duration: {epoch_duration:.1f}s | Total elapsed: {total_elapsed/60:.1f}m"
        )

        # Runtime Check: Ensure we leave ~20 mins for analysis and inference
        # 100 minutes = 6000 seconds
        if total_elapsed > 6000:
            print(
                "Time limit approaching (100 mins exceeded). Stopping training early."
            )
            break

    print(f"Training finished. Best F1 (approx): {best_f1:.5f} at Epoch {best_epoch}")

    # -------------------------------------------------------------------------
    # 6. Evaluation & Threshold Optimization
    # -------------------------------------------------------------------------
    print("Loading best model for final evaluation and threshold tuning...")

    # Load the best saved state
    best_model = ArtworkModel(pretrained=False)
    best_model.to(device)
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    best_model.load_state_dict(checkpoint["state_dict"])
    best_model.eval()

    # Get raw logits for validation set
    _, _, val_logits, val_targets = validate(best_model, val_loader, device)

    # Convert to numpy for fast threshold search
    val_probs = torch.sigmoid(val_logits).cpu().numpy()
    val_targets_np = val_targets.cpu().numpy()

    print("Searching for optimal threshold...")
    best_thresh = 0.5
    final_best_score = 0.0

    thresholds = np.arange(
        Config.THRESHOLD_START, Config.THRESHOLD_END, Config.THRESHOLD_STEP
    )

    for thresh in thresholds:
        preds = (val_probs > thresh).astype(int)
        score = f1_score(val_targets_np, preds, average="micro")
        if score > final_best_score:
            final_best_score = score
            best_thresh = thresh

    print(f"Optimal Threshold: {best_thresh:.2f}")
    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_best_score}")

    # -------------------------------------------------------------------------
    # 7. Failure Analysis
    # -------------------------------------------------------------------------
    print("Performing Failure Analysis...")

    # Calculate error magnitude per sample (1 - F1 score per sample)
    final_preds = (val_probs > best_thresh).astype(int)

    # Vectorized calculation of sample-wise F1
    # F1 = 2TP / (2TP + FP + FN)
    tp = (final_preds * val_targets_np).sum(axis=1)
    fp = (final_preds * (1 - val_targets_np)).sum(axis=1)
    fn = ((1 - final_preds) * val_targets_np).sum(axis=1)

    epsilon = 1e-7
    f1_per_sample = (2 * tp) / (2 * tp + fp + fn + epsilon)
    error_magnitude = 1.0 - f1_per_sample

    # Load metadata to get input features (Number of Labels)
    # We assume val_loader iterates sequentially over Config.VAL_METADATA
    val_df = pd.read_csv(Config.VAL_METADATA, dtype={"attribute_ids": str})

    # Calculate label cardinality for each validation sample
    def get_label_count(x):
        if pd.isna(x) or x == "":
            return 0
        return len(x.split())

    val_df["num_labels"] = val_df["attribute_ids"].apply(get_label_count)
    num_labels = val_df["num_labels"].values

    # Ensure alignment
    if len(num_labels) == len(error_magnitude):
        correlation = np.corrcoef(num_labels, error_magnitude)[0, 1]
        print(
            f"Correlation between Error Magnitude and Number of Labels: {correlation}"
        )
    else:
        print(
            f"Warning: Shape mismatch in failure analysis ({len(num_labels)} vs {len(error_magnitude)}). Skipping correlation."
        )

    # -------------------------------------------------------------------------
    # 8. Submission
    # -------------------------------------------------------------------------
    TARGET_METRIC = 0.6106623748931248

    if final_best_score > TARGET_METRIC:
        print(
            f"Validation metric ({final_best_score}) > Target ({TARGET_METRIC}). Generating submission..."
        )

        test_logits, test_ids = predict(best_model, test_loader, device)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

        submission_rows = []
        for i, img_id in enumerate(test_ids):
            # Select classes above the optimal threshold
            indices = np.where(test_probs[i] > best_thresh)[0]
            attr_str = " ".join(map(str, indices))
            submission_rows.append({"id": img_id, "attribute_ids": attr_str})

        sub_df = pd.DataFrame(submission_rows)
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"Validation metric ({final_best_score}) did not meet target ({TARGET_METRIC}). Submission skipped."
        )


if __name__ == "__main__":
    run()
