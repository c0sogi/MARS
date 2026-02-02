import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.loss import MaskedMCRMSELoss
from library.data import get_loaders
from library.model import ScalePartitionedDenseNet
from library.train import train_epoch, validate, inference, generate_submission

# Suppress warnings
warnings.filterwarnings("ignore")


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes the model's performance on the validation set to identify
    correlations between error and input features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    # 1. Collect per-sample errors
    all_ids = val_loader.dataset.ids
    sample_errors = []

    # Indices for scored columns
    target_cols = Config.TARGET_COLS
    scored_cols = set(Config.SCORED_COLS)
    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]
    seq_scored = Config.SEQ_SCORED

    with torch.no_grad():
        for inputs, partner_indices, targets in val_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            # Forward
            preds = model(inputs, partner_indices)

            # Slice to scored length
            preds = preds[:, :seq_scored, :]
            targets = targets[:, :seq_scored, :]

            # Select scored columns
            preds = preds[:, :, scored_indices]
            targets = targets[:, :, scored_indices]

            # Compute MSE per sample: (Batch, SeqLen, Channels) -> (Batch,)
            # Mean over SeqLen and Channels
            mse_per_sample = torch.mean((preds - targets) ** 2, dim=(1, 2))
            rmse_per_sample = torch.sqrt(mse_per_sample)

            batch_errors = rmse_per_sample.cpu().numpy()
            sample_errors.extend(batch_errors)

    # 2. Load Metadata
    val_csv_path = Config.VAL_CSV
    if not os.path.exists(val_csv_path):
        print("Validation metadata not found. Skipping detailed metadata analysis.")
        return

    df_val = pd.read_csv(val_csv_path)

    # Map errors to IDs
    # val_loader is not shuffled, so order should match, but mapping is safer
    error_map = {id_: err for id_, err in zip(all_ids, sample_errors)}
    df_val["model_error"] = df_val["id"].map(error_map)

    # 3. Feature Engineering for Analysis
    df_val["count_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["count_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["count_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["count_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

    # Features to correlate
    features = [
        "signal_to_noise",
        "SN_filter",
        "seq_length",
        "count_A",
        "count_G",
        "count_C",
        "count_U",
    ]

    # Filter only numeric columns that exist
    features = [f for f in features if f in df_val.columns]

    print(f"{'Feature':<20} | {'Correlation with Error':<25}")
    print("-" * 50)

    for feat in features:
        valid_data = df_val[[feat, "model_error"]].dropna()
        if len(valid_data) > 1:
            corr = np.corrcoef(valid_data[feat], valid_data["model_error"])[0, 1]
            print(f"{feat:<20} | {corr:.4f}")
        else:
            print(f"{feat:<20} | N/A")

    print("-" * 50)


def main():
    # 1. Configuration & Setup
    Config.setup()
    # Fast baseline settings
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 32

    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = ScalePartitionedDenseNet().to(device)

    # 4. Training Setup
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = MaskedMCRMSELoss().to(device)

    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.IDEA_DIR, "best_model.pth")

    # 5. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_mcrmse = validate(model, val_loader, loss_fn, device)

        scheduler.step(val_mcrmse)

        # print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val MCRMSE={val_mcrmse:.4f}")

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Validation & Metric
    print("Training complete. Loading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    val_loss, final_mcrmse = validate(model, val_loader, loss_fn, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mcrmse}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Submission Logic
    THRESHOLD = 0.5417620723771521

    if final_mcrmse < THRESHOLD:
        print(
            f"Metric ({final_mcrmse}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference
        test_preds = inference(model, test_loader, device)
        test_ids = test_loader.dataset.ids

        # Generate Submission
        generate_submission(test_preds, test_ids, Config.SUBMISSION_PATH)
    else:
        print(
            f"Metric ({final_mcrmse}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
