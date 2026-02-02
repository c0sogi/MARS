import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.loss import LaplaceLogLikelihoodLoss
from library.model import OSICModel
from library.data import get_dataloaders
from library.train import train_one_epoch, evaluate
from library.inference import predict_test


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between error magnitude and input features.
    """
    model.eval()
    all_fvc_pred = []
    all_targets = []

    # Run inference on validation set
    with torch.no_grad():
        for imgs, tabs, targets in val_loader:
            imgs = imgs.to(device)
            tabs = tabs.to(device)

            preds = model(imgs, tabs)
            fvc_pred = preds[:, 0].cpu().numpy()

            all_fvc_pred.extend(fvc_pred)
            all_targets.extend(targets.flatten().numpy())

    # Convert to numpy arrays and Unscale
    # Cite {solution_lesson_node_00001}
    y_pred = np.array(all_fvc_pred) * Config.TARGET_STD + Config.TARGET_MEAN
    y_true = np.array(all_targets) * Config.TARGET_STD + Config.TARGET_MEAN

    # Calculate Error Magnitude
    errors = np.abs(y_true - y_pred)

    # Retrieve Validation Metadata to correlate with features
    # val_loader.dataset.data is the dataframe used for the dataset
    val_df = val_loader.dataset.data.copy()

    # Ensure lengths match (loader might drop last batch if configured, but here drop_last is default False)
    if len(errors) != len(val_df):
        print(
            f"Warning: Length mismatch in failure analysis. Errors: {len(errors)}, DF: {len(val_df)}"
        )
        # Truncate to match (safe fallback)
        min_len = min(len(errors), len(val_df))
        errors = errors[:min_len]
        val_df = val_df.iloc[:min_len]

    val_df["Error_Magnitude"] = errors

    # Select numerical features for correlation
    # We look at: Weeks (time), FVC (target/baseline), Percent, Age
    features_to_analyze = [
        "Weeks",
        "FVC",
        "Percent",
        "Age",
        "Baseline_FVC",
        "Baseline_Percent",
    ]
    # Filter for columns that actually exist
    features_to_analyze = [c for c in features_to_analyze if c in val_df.columns]

    print("\n=== Failure Analysis ===")
    print("Correlation between Error Magnitude and Features:")
    correlations = (
        val_df[features_to_analyze + ["Error_Magnitude"]]
        .corr()["Error_Magnitude"]
        .drop("Error_Magnitude")
    )
    print(correlations)
    print("========================\n")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()

    # Fast Baseline Settings
    # We stick to Config defaults mostly, but ensure we don't over-run time.
    # 20 epochs on ~1000 rows is very fast (< 5 mins).
    print(f"Running Fast Baseline on {Config.DEVICE}")

    # 2. Data Loading
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    # 3. Model Initialization
    device = torch.device(Config.DEVICE)
    model = OSICModel().to(device)

    criterion = LaplaceLogLikelihoodLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_metric = -float("inf")  # Metric is negative, higher is better
    # Note: We track loss for early stopping usually, but the prompt emphasizes the metric.
    # The config uses loss for early stopping. We will stick to saving best loss for consistency with library.train.
    best_loss = float("inf")
    best_model_path = os.path.join(Config.MODEL_CHECKPOINT_DIR, "best_model.pth")

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metric = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Save if best loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        # Log progress (optional, kept minimal)
        # print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Metric {val_metric:.4f}")

    print("Training Complete.")

    # 5. Final Validation & Metric Reporting
    print("Loading best model for validation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Re-evaluate on full validation set
    final_val_loss, final_val_metric = evaluate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_metric}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Submission Generation
    # Using the provided inference module which handles the test set expansion and formatting
    if final_val_metric > -6.865544455310175:
        predict_test(checkpoint_path=best_model_path, device=Config.DEVICE)
    else:
        print("Metric not improved. Skipping submission.")


if __name__ == "__main__":
    main()
