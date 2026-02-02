import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr

# Import library components
from library.config import Config
from library.utils import seed_everything, MetricTracker
from library.loss import MaskedMCRMSELoss
from library.data import get_loaders
from library.model import DenseStackingHybridNet
from library.train import train_epoch, validate, generate_submission

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Configuration & Setup
    Config.setup()
    # Override epochs for fast baseline execution
    Config.EPOCHS = 15

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # debug=False ensures we use the full dataset for valid metrics
    train_loader, val_loader, test_loader, test_ids = get_loaders(debug=False)

    # 3. Model Initialization
    model = DenseStackingHybridNet().to(device)

    # 4. Training Components
    criterion = MaskedMCRMSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    # Verbose=False to minimize output
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # 5. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    tracker = MetricTracker()

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, tracker, device)

        # Scheduler step
        scheduler.step(val_mcrmse)

        # Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Evaluation
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Compute final metric on validation set
    _, final_mcrmse = validate(model, val_loader, criterion, tracker, device)
    print(f"Final Validation Metric: {final_mcrmse}")

    # 7. Failure Analysis
    val_df = pd.read_csv(Config.VAL_CSV)

    # Get predictions and targets for validation set
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, neighbor_indices, targets in val_loader:
            inputs = inputs.to(device)
            neighbor_indices = neighbor_indices.to(device)
            outputs = model(inputs, neighbor_indices)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Indices for scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    preds_scored = all_preds[:, :, scored_indices]
    targets_scored = all_targets[:, :, scored_indices]

    # Compute RMSE per sample (averaged over sequence and scored columns)
    # (N, L, 3) -> (N,)
    squared_error = (preds_scored - targets_scored) ** 2
    mse_per_sample = np.mean(squared_error, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Correlate with metadata features
    features = ["signal_to_noise", "mean_reactivity", "SN_filter"]
    for feat in features:
        if feat in val_df.columns:
            vals = val_df[feat].fillna(0).values
            # Ensure alignment (val_loader is sequential, val_df is source)
            if len(vals) == len(rmse_per_sample):
                corr, _ = pearsonr(vals, rmse_per_sample)
                print(f"Correlation between Error and {feat}: {corr:.4f}")

    # 8. Submission
    THRESHOLD = 0.5421870350837708
    if final_mcrmse < THRESHOLD:
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        generate_submission(model, test_loader, test_ids, device, submission_path)


if __name__ == "__main__":
    main()
