import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib

from library.config import (
    SEED,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    WORKING_DIR,
    SUBMISSION_PATH,
    MODEL_FEATURES,
    BATCH_SIZE,
)
from library.data_loader import get_data_loaders
from library.model import KARHNet
from library.engine import train_fn, eval_fn, predict_fn
from library.utils import seed_everything, MaskedMAELoss


def main():
    # 1. Setup
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fast baseline settings (overriding config for speed)
    NUM_EPOCHS = 15

    print(f"Device: {device}")
    print(f"Training for {NUM_EPOCHS} epochs...")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=True)

    # 3. Model & Optimizer
    model = KARHNet().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = MaskedMAELoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_fn(
            model, train_loader, optimizer, device, loss_fn, MAX_GRAD_NORM
        )
        val_loss = eval_fn(model, val_loader, device, loss_fn)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Evaluation & Failure Analysis
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Compute Final Metric
    final_val_loss = eval_fn(model, val_loader, device, loss_fn)
    print(f"Final Validation Metric: {final_val_loss}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    val_errors = []
    val_features = []

    # Collect data for analysis
    # We need to manually iterate to get features and errors corresponding to u_out == 0
    with torch.no_grad():
        for inputs, targets, u_out in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            u_out = u_out.to(device)

            outputs = model(inputs)

            # Calculate absolute error
            error = torch.abs(outputs - targets)

            # Mask: Only inspiratory phase (u_out == 0)
            mask = u_out == 0

            # Filter valid steps
            valid_error = error[mask]
            valid_inputs = inputs[mask]  # Shape: (N_valid, N_features)

            val_errors.append(valid_error.cpu().numpy())
            val_features.append(valid_inputs.cpu().numpy())

    # Concatenate
    all_errors = np.concatenate(val_errors)
    all_features = np.concatenate(val_features)

    # Calculate correlations
    # all_features is (N, n_features), all_errors is (N,)
    print("Correlation between Error Magnitude and Features:")
    feature_corrs = {}
    for i, feature_name in enumerate(MODEL_FEATURES):
        feat_vals = all_features[:, i]
        # Handle potential constant features (std=0) to avoid NaN correlation
        if np.std(feat_vals) > 1e-9:
            corr = np.corrcoef(feat_vals, all_errors)[0, 1]
            feature_corrs[feature_name] = corr
        else:
            feature_corrs[feature_name] = 0.0

    # Sort and print
    sorted_corrs = sorted(feature_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corrs[:10]:  # Print top 10
        print(f"{name}: {corr:.4f}")

    # 6. Submission
    THRESHOLD = 0.16391726930343686
    if final_val_loss < THRESHOLD:
        print(
            f"\nValidation metric {final_val_loss} meets threshold {THRESHOLD}. Generating submission..."
        )

        # Generate predictions
        preds = predict_fn(model, test_loader, device)

        # Load Test IDs
        test_ids_path = os.path.join(WORKING_DIR, "test_ids.npy")
        if os.path.exists(test_ids_path):
            test_ids = np.load(test_ids_path).flatten()
        else:
            # Fallback if cache missing (unlikely given get_data_loaders logic)
            test_df = pd.read_csv("./metadata/test.csv")
            test_ids = test_df["id"].values

        # Ensure shapes match
        if len(preds) != len(test_ids):
            print(
                f"Warning: Prediction length {len(preds)} != ID length {len(test_ids)}"
            )
            # Truncate or pad if necessary, though this indicates a bug
            min_len = min(len(preds), len(test_ids))
            preds = preds[:min_len]
            test_ids = test_ids[:min_len]

        # Create DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "pressure": preds})

        # Save
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {final_val_loss} does not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
