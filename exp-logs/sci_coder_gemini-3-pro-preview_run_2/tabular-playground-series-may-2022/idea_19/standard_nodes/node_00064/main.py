import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from provided library
from library.config import Config
from library.utils import seed_everything, print_metric, calculate_auc
from library.dataset import get_dataloaders
from library.model import NormFusionResFunnel
from library.engine import train_model, predict, evaluate


def failure_analysis(model, val_loader, device):
    """
    Performs failure analysis by correlating prediction errors with input features.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    all_errors = []
    all_feats = []

    # Collect data
    with torch.no_grad():
        for batch in val_loader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            outputs = model(continuous, sequence)
            probs = torch.sigmoid(outputs)

            # Calculate absolute error
            errors = torch.abs(targets - probs)

            all_errors.append(errors.cpu().numpy())
            # We analyze continuous features for correlation
            all_feats.append(continuous.cpu().numpy())

    all_errors = np.concatenate(all_errors).flatten()
    all_feats = np.concatenate(all_feats, axis=0)

    # Create DataFrame for correlation
    feat_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
    df_analysis = pd.DataFrame(all_feats, columns=feat_cols)
    df_analysis["error"] = all_errors

    # Compute correlation
    correlations = (
        df_analysis.corrwith(df_analysis["error"]).abs().sort_values(ascending=False)
    )

    print("Top 5 Features correlated with Error Magnitude:")
    print(correlations.head(6))  # head(6) because 'error' itself will be 1.0
    return correlations


def main():
    # 1. Setup
    # Override Config for this run
    Config.NUM_EPOCHS = (
        20  # Sufficient for convergence on A100, keeps it relatively fast
    )
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing run on {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = NormFusionResFunnel().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    # 5. Training
    print("Starting training...")
    # train_model returns the best AUC achieved during training
    _ = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # 6. Validation Assessment
    print("\n--- Final Validation Assessment ---")
    # Load best model
    checkpoint = torch.load(Config.MODEL_PATH)
    model.load_state_dict(checkpoint)

    # Evaluate
    criterion = nn.BCEWithLogitsLoss()
    val_loss, val_auc = evaluate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    failure_analysis(model, val_loader, device)

    # 8. Submission
    THRESHOLD = 0.9970005855169476

    if val_auc > THRESHOLD:
        print(f"\nValidation metric {val_auc} > {THRESHOLD}. Generating submission...")
        predict(model, test_loader, device)
    else:
        print(
            f"\nValidation metric {val_auc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
