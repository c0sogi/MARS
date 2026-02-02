import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import provided library modules
from library.config import Config
from library.utils import seed_all, MaskedMCRMSELoss
from library.data import get_dataloaders
from library.model import DensePartnerAwareNet
from library.train import train_one_epoch, evaluate, generate_submission


def main():
    # 1. Setup
    seed_all(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override Config for Fast Baseline execution
    Config.EPOCHS = 15

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=False, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = DensePartnerAwareNet(
        input_channels=Config.INPUT_CHANNELS,
        tcn_channels=Config.TCN_CHANNELS,
        tcn_layers=Config.TCN_LAYERS,
        kernel_size=Config.TCN_KERNEL_SIZE,
        dropout=Config.DROPOUT,
        latent_dim=Config.LATENT_DIM,
        gru_hidden=Config.GRU_HIDDEN_DIM,
        num_targets=Config.NUM_TARGETS,
    ).to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=False
    )
    criterion = MaskedMCRMSELoss(scored_indices=Config.SCORED_INDICES)

    # 5. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_mcrmse = float("inf")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_mcrmse = evaluate(model, val_loader, device)

        scheduler.step(val_mcrmse)

        # Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 6. Final Evaluation
    print("Training complete. Loading best model...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    final_val_metric = evaluate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_metric}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()
    val_errors = []

    # Compute error per sample
    scored_indices = Config.SCORED_INDICES

    with torch.no_grad():
        for inputs, partner_indices, targets in val_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            preds = model(inputs, partner_indices)

            # Filter scored columns
            p_scored = preds[:, :, scored_indices]
            t_scored = targets[:, :, scored_indices]

            # Compute MCRMSE per sample
            # MSE per position per column: (B, L, 3)
            mse = (p_scored - t_scored) ** 2
            # Mean MSE per column over sequence: (B, 3)
            mse_per_col = torch.mean(mse, dim=1)
            # RMSE per column: (B, 3)
            rmse_per_col = torch.sqrt(mse_per_col + 1e-8)
            # Mean RMSE across columns (MCRMSE): (B,)
            mcrmse_per_sample = torch.mean(rmse_per_col, dim=1)

            val_errors.extend(mcrmse_per_sample.cpu().numpy())

    # Load metadata to correlate
    val_df = pd.read_csv(Config.VAL_DATA_PATH)

    # Assign errors to dataframe (order is preserved in val_loader)
    # val_loader.dataset.ids matches the order of rows in val_df if not shuffled
    val_df["model_error"] = val_errors

    # Feature Engineering for Analysis
    val_df["count_A"] = val_df["sequence"].apply(lambda x: x.count("A"))
    val_df["count_G"] = val_df["sequence"].apply(lambda x: x.count("G"))
    val_df["count_C"] = val_df["sequence"].apply(lambda x: x.count("C"))
    val_df["count_U"] = val_df["sequence"].apply(lambda x: x.count("U"))

    analysis_features = ["signal_to_noise", "count_A", "count_G", "count_C", "count_U"]
    correlations = (
        val_df[analysis_features + ["model_error"]]
        .corr()["model_error"]
        .drop("model_error")
    )

    print("Correlation between Input Features and Model Error:")
    print(correlations)

    # 8. Submission Generation
    THRESHOLD = 0.6477736930052439
    if final_val_metric < THRESHOLD:
        print(
            f"\nValidation metric meets threshold ({final_val_metric} < {THRESHOLD}). Generating submission..."
        )

        # Create output directory
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        generate_submission(model, test_loader, device, submission_path)
    else:
        print(
            f"\nValidation metric {final_val_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
