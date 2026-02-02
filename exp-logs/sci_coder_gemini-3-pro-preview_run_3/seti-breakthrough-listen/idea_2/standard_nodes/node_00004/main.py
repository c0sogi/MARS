import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from library.config import Config, set_seed, setup_directories
from library.dataset import get_dataloaders
from library.model import SETIEfficientNet
from library.engine import fit, predict_and_submit
from library.utils import get_score


def main():
    # --- 1. Setup & Configuration ---
    setup_directories()
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Using device: {device}")

    # --- 2. Data Preparation (Fast Baseline) ---
    # We subsample the training data to ensure the script completes quickly.
    full_train_df = pd.read_csv(Config.TRAIN_CSV)
    MAX_TRAIN_SAMPLES = 15000

    if len(full_train_df) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data to {MAX_TRAIN_SAMPLES} samples for fast baseline..."
        )
        train_subset = full_train_df.sample(
            n=MAX_TRAIN_SAMPLES, random_state=Config.SEED
        ).reset_index(drop=True)
        temp_train_csv = os.path.join(Config.WORKING_DIR, "train_subset.csv")
        train_subset.to_csv(temp_train_csv, index=False)
        train_csv_path = temp_train_csv
    else:
        train_csv_path = Config.TRAIN_CSV

    # Initialize DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv=train_csv_path,
        val_csv=Config.VAL_CSV,
        test_csv=Config.TEST_CSV,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # --- 3. Model Initialization ---
    model = SETIEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    )
    model = model.to(device)

    # --- 4. Training Configuration ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = nn.BCEWithLogitsLoss()

    # --- 5. Training Loop ---
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=Config.NUM_EPOCHS,
        patience=3,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # --- 6. Validation & Failure Analysis ---
    print("\nLoading best model for analysis...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    val_probs = []
    val_targets = []
    meta_stats = []

    print("Running validation inference and extracting signal statistics...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            # Inference
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            val_probs.extend(probs)
            val_targets.extend(targets.cpu().numpy().flatten())

            # Extract statistics for failure analysis
            # images shape: (B, 6, H, W)
            # On-target indices: 0, 2, 4 | Off-target indices: 1, 3, 5
            imgs_np = images.cpu().numpy()
            for i in range(imgs_np.shape[0]):
                img = imgs_np[i]
                on_target = img[[0, 2, 4], :, :]
                off_target = img[[1, 3, 5], :, :]

                stats = {
                    "mean_on": np.mean(on_target),
                    "std_on": np.std(on_target),
                    "max_on": np.max(on_target),
                    "mean_off": np.mean(off_target),
                    "std_off": np.std(off_target),
                    "max_off": np.max(off_target),
                    "mean_diff": np.mean(on_target) - np.mean(off_target),
                }
                meta_stats.append(stats)

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    # Calculate and Print Metric
    final_auc = get_score(val_targets, val_probs)
    print(f"Final Validation Metric: {final_auc}")

    # Perform Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(val_targets - val_probs)
    df_stats = pd.DataFrame(meta_stats)
    df_stats["error"] = errors

    # Calculate correlations
    correlations = df_stats.corr()["error"].drop("error").sort_values(ascending=False)
    print("Correlation between Error Magnitude and Signal Features:")
    print(correlations)

    # --- 7. Submission ---
    THRESHOLD = 0.5170457784564271
    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation AUC ({final_auc}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
