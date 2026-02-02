import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.stats as stats

# Import provided library modules
from library import config, utils, loss, data, model, train

# Override configuration for fast baseline execution
config.NUM_EPOCHS = 20


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = utils.get_device()

    # 2. Data Loading
    train_loader, val_loader, test_loader = data.get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    net = model.HCHSGFN().to(device)

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = loss.MaskedMCRMSELoss().to(device)

    # 5. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    for epoch in range(config.NUM_EPOCHS):
        # Train
        _ = train.train_one_epoch(net, train_loader, optimizer, criterion, device)

        # Validate
        val_score = train.validate(net, val_loader, device)

        # Scheduler Step
        scheduler.step(val_score)

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            torch.save(net.state_dict(), best_model_path)

    # 6. Final Evaluation
    # Load best model
    net.load_state_dict(torch.load(best_model_path, map_location=device))

    # Compute final metric
    final_val_score = train.validate(net, val_loader, device)
    print(f"Final Validation Metric: {final_val_score}")

    # 7. Failure Analysis
    perform_failure_analysis(net, val_loader, device)

    # 8. Conditional Submission
    threshold = 0.47142532743789534
    if final_val_score < threshold:
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        train.generate_submission(net, test_loader, device, submission_path)


def perform_failure_analysis(model_instance, val_loader, device):
    """
    Analyzes model performance on the validation set by correlating
    error magnitude with input features.
    """
    model_instance.eval()
    all_preds = []
    all_targets = []

    # Inference on validation set
    with torch.no_grad():
        for x, p_idx, y in val_loader:
            x = x.to(device)
            p_idx = p_idx.to(device)

            # Pass 1
            pred1 = model_instance(x, p_idx, y_prev=None)
            # Pass 2 (Final)
            pred2 = model_instance(x, p_idx, y_prev=pred1)

            all_preds.append(pred2.cpu().numpy())
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Calculate per-sample RMSE on scored columns and positions
    # Slice to valid scored region
    preds_valid = preds[:, : config.SEQ_SCORED, :][:, :, config.SCORED_COLS_INDICES]
    targets_valid = targets[:, : config.SEQ_SCORED, :][:, :, config.SCORED_COLS_INDICES]

    # MSE per sample (average over positions and columns)
    mse_per_sample = np.mean((preds_valid - targets_valid) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load Metadata
    val_csv_path = os.path.join(config.METADATA_DIR, "val.csv")
    if not os.path.exists(val_csv_path):
        print("Validation metadata not found, skipping failure analysis.")
        return

    df_val = pd.read_csv(val_csv_path)
    df_val["rmse_error"] = rmse_per_sample

    # Feature Engineering for Analysis
    df_val["count_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["count_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["count_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["count_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

    features_to_check = [
        "signal_to_noise",
        "SN_filter",
        "count_A",
        "count_G",
        "count_C",
        "count_U",
    ]

    print("Failure Analysis (Correlation between Error and Features):")
    for feat in features_to_check:
        if feat in df_val.columns:
            # Check if feature has variance
            if df_val[feat].nunique() > 1:
                corr, _ = stats.pearsonr(df_val[feat], df_val["rmse_error"])
                print(f"Correlation Error vs {feat}: {corr:.6f}")


if __name__ == "__main__":
    main()
