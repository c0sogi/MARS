import sys
import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.loss import MaskedL1Loss
from library.model import PCDRHNet
from library.train import load_data_and_create_loaders, train_epoch, validate, predict


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Full Training Configuration
    # Using Config.EPOCHS (60) to ensure convergence (Cite 00054, 00071)
    EPOCHS = Config.EPOCHS

    # 2. Data Loading
    # Using cached data if available as per instructions
    train_loader, val_loader, test_loader, test_df = load_data_and_create_loaders()

    # 3. Model Initialization
    input_dim = len(Config.STREAM_A_FEATURES)
    model = PCDRHNet(input_dim).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    criterion = MaskedL1Loss()

    # 4. Training Loop
    best_loss = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model_baseline.pth")

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Validation Metric
    # Load best model for accurate metric reporting
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    final_val_loss = validate(model, val_loader, criterion, device)

    print(f"Final Validation Metric: {final_val_loss}")

    # 6. Failure Analysis
    model.eval()
    all_errors = []
    all_features = []

    # We need to correlate errors with input features.
    # We will iterate over validation set, compute error, and collect features.

    with torch.no_grad():
        for x_a, x_b, y in val_loader:
            x_a = x_a.to(device)
            x_b = x_b.to(device)
            y = y.to(device)

            pred = model(x_a)

            # Compute absolute error
            abs_error = torch.abs(pred - y)

            # Masking: We only care about inspiratory phase (u_out == 0)
            # x_b is (Batch, 80, 1), squeeze to (Batch, 80)
            u_out = x_b.squeeze(-1)
            mask = (1.0 - u_out) > 0.5  # Boolean mask where u_out is 0

            # Flatten tensors
            mask_flat = mask.view(-1)

            # Filter error and features using the mask
            valid_errors = abs_error.view(-1)[mask_flat]
            valid_features = x_a.view(-1, input_dim)[mask_flat]

            all_errors.append(valid_errors.cpu().numpy())
            all_features.append(valid_features.cpu().numpy())

    # Concatenate results
    if len(all_errors) > 0:
        all_errors = np.concatenate(all_errors)
        all_features = np.concatenate(all_features, axis=0)

        # Create DataFrame for correlation
        analysis_df = pd.DataFrame(all_features, columns=Config.STREAM_A_FEATURES)
        analysis_df["error"] = all_errors

        # Compute correlation
        correlations = analysis_df.corr()["error"].drop("error")
        print("Correlation between Error and Features:")
        print(correlations.sort_values(ascending=False))
    else:
        print("No validation samples found for analysis.")

    # 7. Conditional Submission
    THRESHOLD = 0.16391726930343686

    if final_val_loss < THRESHOLD:
        # Generate predictions
        # predict function returns (N_breaths, 80) numpy array
        predictions = predict(model, test_loader, device)

        # Flatten to match submission format
        predictions_flat = predictions.flatten()

        # Create submission DataFrame
        # test_df was returned by load_data_and_create_loaders and preserves order
        submission = pd.DataFrame(
            {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: predictions_flat}
        )

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
    else:
        pass


if __name__ == "__main__":
    main()
