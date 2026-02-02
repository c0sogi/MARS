import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import from provided libraries
from library.config import Config, set_seed
from library.utils import get_device, clean_cache
from library.dataset import prepare_data
from library.model import FCPNet
from library.train import train_epoch, validate, generate_submission


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Clean cache to ensure fresh data processing
    clean_cache()

    # Set reproducibility
    set_seed(Config.SEED)

    # Device selection
    device = get_device()
    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    print("Preparing data...")
    # load_cached_data=True allows using the cache generated in this run if we re-ran parts,
    # but since we cleaned cache, it will generate fresh data.
    train_loader, val_loader, test_loader = prepare_data(
        debug=False, load_cached_data=True
    )

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("Initializing FCP-Net...")
    model = FCPNet(config=Config).to(device)

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.FACTOR,
        patience=5,
        min_lr=Config.MIN_LR,
    )

    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_val_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Scheduler update
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            # print(f"Saved best model (Loss: {best_val_loss:.6f})")

    # ---------------------------------------------------------
    # 5. Final Evaluation
    # ---------------------------------------------------------
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Compute final metric on the hold-out validation set
    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_errors = []
    all_features = []

    # u_out is at index 2 in Config.FEATURE_COLS
    U_OUT_IDX = Config.FEATURE_COLS.index("u_out")

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)  # (B, Seq, 1)

            # Flatten for analysis
            inputs_flat = inputs.reshape(-1, inputs.shape[-1])
            targets_flat = targets.reshape(-1)
            outputs_flat = outputs.reshape(-1)

            # Mask: Only analyze inspiratory phase (u_out == 0)
            mask = inputs_flat[:, U_OUT_IDX] == 0

            if mask.sum() > 0:
                # Calculate absolute error
                errors = torch.abs(outputs_flat[mask] - targets_flat[mask])
                features = inputs_flat[mask]

                all_errors.append(errors.cpu().numpy())
                all_features.append(features.cpu().numpy())

    # Concatenate results
    if all_errors:
        all_errors = np.concatenate(all_errors)
        all_features = np.concatenate(all_features)

        # Create DataFrame for correlation
        df_analysis = pd.DataFrame(all_features, columns=Config.FEATURE_COLS)
        df_analysis["error"] = all_errors

        # Compute correlation
        print("Correlation between Error Magnitude and Features:")
        correlations = df_analysis.corr()["error"].sort_values(ascending=False)
        print(correlations)
    else:
        print("No inspiratory phase data found for analysis.")

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.23978149890899658

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
