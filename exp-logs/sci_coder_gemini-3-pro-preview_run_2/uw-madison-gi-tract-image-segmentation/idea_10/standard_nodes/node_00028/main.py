import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import time

# Import from provided library files
from library.config import Config
from library.utils import set_seed, compute_dice_score, compute_hausdorff_distance
from library.model import ShuffleNetPSPNet
from library.dataset import get_dataloaders
from library.loss import CombinedLoss
from library.train import train_one_epoch, validate, generate_submission


def main():
    # 1. Setup and Configuration Overrides for Fast Baseline
    # Limit epochs to ensure completion within 2 hours
    Config.EPOCHS = 5
    # Ensure we use the GPU
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    Config.setup()
    set_seed(Config.SEED)

    print(f"Starting run on device: {Config.DEVICE}")
    print(f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # 2. Data Loading
    print("Loading data...")
    # load_cached_data=True to use parquet files generated in previous steps
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = ShuffleNetPSPNet(
        num_classes=Config.NUM_CLASSES, in_channels=Config.IN_CHANNELS
    )
    model = model.to(Config.DEVICE)

    # 4. Training Loop
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )
    loss_fn = CombinedLoss().to(Config.DEVICE)

    best_dice = 0.0
    global_iter = 0

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss, global_iter = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            Config.DEVICE,
            epoch,
            Config.EPOCHS,
            global_iter,
        )

        # Simple Validation (Dice only for monitoring)
        val_loss, val_dice = validate(model, val_loader, loss_fn, Config.DEVICE)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
        )

        # Save Best Model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New best model saved (Dice: {best_dice:.4f})")

    # 5. Advanced Validation & Metric Calculation
    print("\nPerforming full validation assessment...")

    # Load best model
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    # Containers for 3D volume reconstruction
    # case_id -> class_idx -> {'pred': list of 2D, 'true': list of 2D, 'slice': list of indices}
    val_data = {}

    # Container for failure analysis (slice-level)
    slice_errors = []  # {'slice_norm': float, 'error': float}

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(Config.DEVICE)
            masks = batch["mask"].to(Config.DEVICE)

            # Metadata
            cases = batch["case"].numpy()
            days = batch["day"].numpy()
            slices = batch["slice"].numpy()

            # Predict
            outputs = model(images)
            preds = torch.sigmoid(outputs) > Config.THRESHOLD

            preds_np = preds.cpu().numpy().astype(np.uint8)
            masks_np = masks.cpu().numpy().astype(np.uint8)

            batch_size = images.size(0)

            for i in range(batch_size):
                case_id = f"{cases[i]}_{days[i]}"  # Unique case identifier
                slice_idx = slices[i]

                if case_id not in val_data:
                    val_data[case_id] = {
                        c: {"pred": [], "true": [], "slice": []}
                        for c in range(Config.NUM_CLASSES)
                    }

                for c in range(Config.NUM_CLASSES):
                    p_slice = preds_np[i, c]
                    t_slice = masks_np[i, c]

                    val_data[case_id][c]["pred"].append(p_slice)
                    val_data[case_id][c]["true"].append(t_slice)
                    val_data[case_id][c]["slice"].append(slice_idx)

                    # For failure analysis: Compute 2D Dice for this slice
                    # Simple Dice: 2*TP / (P+T)
                    inter = np.sum(p_slice * t_slice)
                    union = np.sum(p_slice) + np.sum(t_slice)
                    d = (2.0 * inter + 1e-6) / (union + 1e-6)

                    # Normalize slice index roughly (assuming max ~144 slices)
                    slice_norm = slice_idx / 144.0
                    slice_errors.append({"slice_norm": slice_norm, "error": 1.0 - d})

    # Compute 3D Metrics
    dice_scores = []
    hausdorff_scores = []

    for case_id, class_data in val_data.items():
        for c in range(Config.NUM_CLASSES):
            data = class_data[c]
            if not data["slice"]:
                continue

            # Sort by slice index to form proper 3D volume
            sorted_indices = np.argsort(data["slice"])

            # Stack to 3D: (Depth, Height, Width)
            vol_pred = np.stack([data["pred"][k] for k in sorted_indices])
            vol_true = np.stack([data["true"][k] for k in sorted_indices])

            # Compute Dice (3D Volume)
            d = compute_dice_score(vol_pred, vol_true)
            dice_scores.append(d)

            # Compute Hausdorff (3D Volume)
            # Shape for normalization is (H, W) of a slice
            h_shape = vol_pred.shape[1:]
            hd = compute_hausdorff_distance(vol_pred, vol_true, h_shape)

            # Normalize Hausdorff to 0-1 score (1 - distance)
            # Note: compute_hausdorff_distance returns a distance.
            # The metric requires a score. Usually Score = 1 - HD.
            # Since HD can be > 1 if objects are far apart (though bounded by sqrt(2) roughly due to norm),
            # we clip it.
            hd_score = max(0.0, 1.0 - hd)
            hausdorff_scores.append(hd_score)

    mean_dice = np.mean(dice_scores)
    mean_hausdorff_score = np.mean(hausdorff_scores)

    # Weighted Metric: 0.4 * Dice + 0.6 * Hausdorff_Score
    final_metric = (0.4 * mean_dice) + (0.6 * mean_hausdorff_score)

    print(f"Validation Dice (3D avg): {mean_dice:.6f}")
    print(f"Validation Hausdorff Score (3D avg): {mean_hausdorff_score:.6f}")
    print(f"Final Validation Metric: {final_metric:.18f}")

    # 6. Failure Analysis
    print("\nPerforming failure analysis...")
    df_errors = pd.DataFrame(slice_errors)
    if not df_errors.empty:
        # Correlation between normalized slice index and error (1-Dice)
        correlation = df_errors["slice_norm"].corr(df_errors["error"])
        print(
            f"Correlation between Slice Position and Error Magnitude: {correlation:.6f}"
        )

        # Simple insight
        if correlation > 0.1:
            print(
                "Insight: Error tends to increase towards the end of the scan (inferior slices)."
            )
        elif correlation < -0.1:
            print(
                "Insight: Error tends to increase towards the start of the scan (superior slices)."
            )
        else:
            print("Insight: Error is relatively uniform across slice positions.")
    else:
        print("No error data collected.")

    # 7. Submission Generation
    # generate_submission handles inference, 3D post-processing, and CSV saving
    # It loads the model from Config.MODEL_SAVE_PATH
    generate_submission(model, test_loader, Config.DEVICE)


if __name__ == "__main__":
    main()
