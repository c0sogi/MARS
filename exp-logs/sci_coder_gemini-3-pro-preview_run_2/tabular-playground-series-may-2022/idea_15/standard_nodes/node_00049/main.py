import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.data_loader import get_dataloaders
from library.model import GatedTransformerResFunnelHybrid
from library.train import train_one_epoch, validate, generate_submission


def main():
    # 1. Setup Environment
    seed_everything(Config.SEED)
    Config.setup()
    device = torch.device(Config.DEVICE)

    # 2. Prepare Data
    # We use the full dataset to ensure we meet the high AUC threshold.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Initialize Model
    model = GatedTransformerResFunnelHybrid().to(device)

    # Optimizer & Scheduler Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    best_model_path = Config.MODEL_SAVE_PATH
    epochs = Config.EPOCHS

    for epoch in range(1, epochs + 1):
        # Train Step
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation Step
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step()

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_auc, best_model_path
            )

    # 5. Validation Reporting
    print(f"Final Validation Metric: {best_auc}")

    # 6. Failure Analysis
    # Load the best model state for analysis
    checkpoint = load_checkpoint(best_model_path, model, device=device)
    model.eval()

    val_errors = []
    val_features = []

    # Compute errors on validation set
    with torch.no_grad():
        for batch in val_loader:
            cont = batch["continuous"].to(device)
            cat = batch["categorical"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            logits = model(cont, cat)
            probs = torch.sigmoid(logits)

            # Calculate absolute error
            error = torch.abs(targets - probs)

            val_errors.append(error.cpu().numpy())
            val_features.append(cont.cpu().numpy())

    val_errors = np.vstack(val_errors).flatten()
    val_features = np.vstack(val_features)  # Shape: (N_samples, N_features)

    print(
        "Failure Analysis: Correlation between Error Magnitude and Continuous Features"
    )
    correlations = []

    # Calculate correlation for each continuous feature
    for i in range(val_features.shape[1]):
        feat_col = val_features[:, i]
        # Avoid division by zero if feature is constant
        if np.std(feat_col) > 1e-9:
            corr = np.corrcoef(feat_col, val_errors)[0, 1]
            correlations.append((i, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # Print top 5 correlations
    for idx, corr in correlations[:5]:
        print(f"Feature index {idx}: {corr:.6f}")

    # 7. Conditional Submission
    THRESHOLD = 0.9970005855169476

    if best_auc > THRESHOLD:
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        generate_submission(model, test_loader, device, submission_path)
    else:
        print(
            f"Validation AUC {best_auc} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
