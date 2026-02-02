import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import hashlib

# Import library modules
from library.config import Config
from library.data_processing import get_data_loaders
from library.model import VentilatorNet
from library.train_eval import train_one_epoch, evaluate, predict, MaskedL1Loss


def main():
    # 1. Configuration Adjustments
    # Use 30 epochs for full convergence (Benchmark idea_10 used 25+)
    Config.EPOCHS = 30
    # Ensure we use the GPU
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running experiment: {Config.EXPERIMENT_NAME}")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {Config.EPOCHS}")

    # 2. Data Loading
    # Load cached data if available, otherwise process from scratch
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=True)

    # 3. Model Initialization
    device = torch.device(Config.DEVICE)
    model = VentilatorNet().to(device)

    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.SCHEDULER_PCT_START,
    )

    criterion = MaskedL1Loss()

    # 5. Training Loop
    best_mae = float("inf")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_mae = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.9f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print(f"Training finished. Best Val MAE: {best_mae:.9f}")

    # 6. Final Evaluation & Failure Analysis
    print("Loading best model for analysis...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Collect validation predictions and targets for analysis
    val_preds = []
    val_targets = []
    val_u_outs = []
    val_inputs = []

    # We need to extract features for correlation analysis
    # Features in order (from data_processing.py):
    # 0:time_step, 1:u_in, 2:R, 3:C, 4:volume, 5:R_u_in, 6:vol_C,
    # 7:u_in_lag1, 8:u_in_lag2, 9:u_in_lag3, 10:u_in_lag4, 11:u_in_diff1, 12:u_in_diff2
    feature_names = [
        "time_step",
        "u_in",
        "R",
        "C",
        "volume",
        "R_u_in",
        "vol_C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_diff1",
        "u_in_diff2",
        "u_out",
    ]

    with torch.no_grad():
        for x, u_out, y in val_loader:
            x = x.to(device)
            u_out_dev = u_out.to(device)

            # Predict
            pred = model(x, u_out_dev)

            # Store data (move to CPU to save GPU memory)
            val_preds.append(pred.cpu().numpy())
            val_targets.append(y.numpy())
            val_u_outs.append(u_out.numpy())
            val_inputs.append(x.cpu().numpy())

    # Concatenate
    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()
    val_u_outs = np.concatenate(val_u_outs).flatten()
    val_inputs = np.concatenate(val_inputs).reshape(-1, len(feature_names))

    # Calculate Metric
    # Filter for inspiratory phase (u_out == 0)
    insp_mask = val_u_outs == 0

    if insp_mask.sum() == 0:
        print("Warning: No inspiratory phase data found in validation set.")
        final_metric = 0.0
    else:
        errors = np.abs(val_preds - val_targets)
        final_metric = np.mean(errors[insp_mask])

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n=== Failure Analysis (Correlation with Error) ===")
    if insp_mask.sum() > 0:
        insp_errors = errors[insp_mask]
        insp_inputs = val_inputs[insp_mask]

        correlations = {}
        for i, name in enumerate(feature_names):
            feat_vals = insp_inputs[:, i]
            # Handle constant features to avoid warning
            if np.std(feat_vals) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(feat_vals, insp_errors)[0, 1]
            correlations[name] = corr

        # Sort by absolute correlation
        sorted_corrs = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )

        print("Feature correlations with Absolute Error (Inspiratory Phase):")
        for name, corr in sorted_corrs:
            print(f"  {name}: {corr:.4f}")
    else:
        print("Skipping failure analysis due to lack of inspiratory data.")

    # 7. Conditional Submission
    TARGET_METRIC = 0.2164510190486908

    if final_metric < TARGET_METRIC:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({TARGET_METRIC}). Generating submission..."
        )

        # Generate predictions
        test_preds = predict(model, test_loader, device)

        # Retrieve Test IDs (Logic replicated from library/train_eval.py)
        # We need to regenerate the hash to find the correct file
        feature_version = "v2_physics_robust_with_uout"
        cache_hash = hashlib.md5(
            f"{feature_version}_{Config.DEBUG}_{Config.EXPERIMENT_NAME}".encode()
        ).hexdigest()
        test_ids_path = os.path.join(Config.CACHE_DIR, f"test_ids_{cache_hash}.npy")

        if not os.path.exists(test_ids_path):
            print(f"Error: Cached test IDs not found at {test_ids_path}")
            return

        test_ids = np.load(test_ids_path).flatten()

        if len(test_ids) != len(test_preds):
            print(
                f"Error: Shape mismatch. IDs: {len(test_ids)}, Preds: {len(test_preds)}"
            )
            return

        submission = pd.DataFrame({"id": test_ids, "pressure": test_preds})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({TARGET_METRIC}). Skipping submission."
        )


if __name__ == "__main__":
    main()
