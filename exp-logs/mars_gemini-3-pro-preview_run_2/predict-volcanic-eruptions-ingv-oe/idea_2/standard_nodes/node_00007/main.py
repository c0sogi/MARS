import os
import sys
import warnings
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library components
from library.config import Config
from library.utils import seed_everything, TargetScaler
from library.dataset import get_dataloaders
from library.model import HybridModel
from library.train import train_one_epoch, validate, generate_submission


def run():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for optimized execution
    # Increasing epochs to allow convergence (Cite solution_lesson_node_00003)
    Config.EPOCHS = 40

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    # load_cached_data=True attempts to reuse features in ./working/idea_2
    # If not found, they will be computed from scratch.
    loaders = get_dataloaders(load_cached_data=False)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # Determine the number of tabular features from the dataset
    # The dataset returns (spectrogram, tabular_features, target)
    sample_spec, sample_tab, _ = train_loader.dataset[0]
    num_tabular_features = sample_tab.shape[0]

    # Load the TargetScaler (fitted and saved during get_dataloaders)
    target_scaler = TargetScaler()
    target_scaler.load(Config.TARGET_MEAN_PATH, Config.TARGET_STD_PATH)

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    model = HybridModel(num_tabular_features=num_tabular_features)
    model = model.to(device)

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    criterion = nn.L1Loss()  # MAE on scaled targets
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    best_val_mae = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train for one epoch
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, device)

        # Validate
        val_loss, val_mae = validate(
            val_loader, model, criterion, target_scaler, device
        )

        # Save Best Model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # ---------------------------------------------------------
    # 5. Validation Assessment
    # ---------------------------------------------------------
    # Load the best saved model state
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Compute final metric on the full validation set
    _, final_mae = validate(val_loader, model, criterion, target_scaler, device)

    # Print the required metric format
    print(f"Final Validation Metric: {final_mae}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    model.eval()
    errors = []
    feature_data = []

    # Collect predictions and features for correlation analysis
    with torch.no_grad():
        for spec, tab, target in val_loader:
            spec = spec.to(device)
            tab = tab.to(device)
            target = target.to(device)

            preds = model(spec, tab)

            # Inverse transform to get original scale
            preds_unscaled = target_scaler.inverse_transform(preds.cpu().numpy())
            target_unscaled = target_scaler.inverse_transform(target.cpu().numpy())

            # Calculate absolute error
            batch_errors = np.abs(preds_unscaled - target_unscaled)
            errors.extend(batch_errors)

            # Collect tabular features (move to CPU)
            feature_data.extend(tab.cpu().numpy())

    errors = np.array(errors)
    feature_data = np.array(feature_data)

    # Create DataFrame for analysis
    feature_names = val_loader.dataset.feature_cols
    df_analysis = pd.DataFrame(feature_data, columns=feature_names)
    df_analysis["error_magnitude"] = errors

    # Calculate correlations between features and error magnitude
    correlations = (
        df_analysis.corr()["error_magnitude"].abs().sort_values(ascending=False)
    )

    print("Failure Analysis - Top Feature Correlations with Error:")
    # Display top 10 correlations (excluding the error column itself)
    print(correlations.drop("error_magnitude", errors="ignore").head(10))

    # ---------------------------------------------------------
    # 7. Submission Logic
    # ---------------------------------------------------------
    THRESHOLD = 4285404.11

    if final_mae < THRESHOLD:
        generate_submission(
            test_loader, model, target_scaler, device, Config.SUBMISSION_PATH
        )


if __name__ == "__main__":
    run()
