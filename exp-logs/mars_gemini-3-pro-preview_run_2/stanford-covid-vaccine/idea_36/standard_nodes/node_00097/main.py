import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import from provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data import get_loader
from library.model import CF_DCN
from library.train import train_one_epoch, validate

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loaders
    # Load training and validation data
    train_loader = get_loader(mode="train", debug=False, load_cached_data=True)
    val_loader = get_loader(mode="val", debug=False, load_cached_data=True)

    # 3. Model & Optimizer
    model = CF_DCN().to(device)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # 4. Training Loop
    # Fast baseline configuration
    num_epochs = 15
    best_val_loss = float("inf")

    # Ensure directory for checkpoints exists
    os.makedirs(os.path.dirname(Config.BEST_MODEL_PATH), exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 5. Final Evaluation
    # Load best model weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    model.eval()
    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Calculate per-sample error
    sample_errors = []
    scored_indices = Config.SCORED_TARGET_INDICES

    # Iterate through validation loader to compute errors
    # Note: val_loader is not shuffled, so order matches val_df
    with torch.no_grad():
        for inputs, partner_indices, targets in val_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            preds = model(inputs, partner_indices)

            # Select scored columns
            preds_scored = preds[:, :, scored_indices]
            targets_scored = targets[:, :, scored_indices]

            # Compute RMSE per sample (averaged over sequence and scored channels)
            # Shape: (Batch, Seq, Channels) -> (Batch,)
            mse = torch.mean((preds_scored - targets_scored) ** 2, dim=(1, 2))
            rmse = torch.sqrt(mse)

            sample_errors.extend(rmse.cpu().numpy())

    val_df["rmse_error"] = sample_errors

    # Compute correlations
    features_to_check = ["signal_to_noise", "mean_reactivity", "SN_filter"]
    print("Correlation between Error and Metadata features:")

    for feat in features_to_check:
        if feat in val_df.columns:
            # Drop NaNs for robust correlation
            subset = val_df[[feat, "rmse_error"]].dropna()

            if len(subset) > 1 and subset[feat].std() > 0:
                corr = np.corrcoef(subset[feat], subset["rmse_error"])[0, 1]
                print(f"{feat}: {corr:.4f}")
            else:
                print(f"{feat}: N/A")

    # 7. Conditional Submission
    THRESHOLD = 0.5403054356575012

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")

        # Create submission directory
        os.makedirs("./submission", exist_ok=True)

        # Load test data
        test_loader = get_loader(mode="test", load_cached_data=True)
        all_preds = []

        with torch.no_grad():
            for inputs, partner_indices, _ in test_loader:
                inputs = inputs.to(device)
                partner_indices = partner_indices.to(device)

                preds = model(inputs, partner_indices)
                all_preds.append(preds.cpu().numpy())

        # Concatenate all predictions: (Total_Samples, Seq_Len, 5)
        all_preds = np.concatenate(all_preds, axis=0)

        # Retrieve IDs from the dataset
        dataset_ids = test_loader.dataset.ids

        submission_data = []
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        # Format predictions
        for i, sample_id in enumerate(dataset_ids):
            sample_preds = all_preds[i]  # Shape (107, 5)
            for seq_pos in range(Config.SEQ_LENGTH):
                row_id = f"{sample_id}_{seq_pos}"
                row_vals = sample_preds[seq_pos]

                row_dict = {"id_seqpos": row_id}
                for col_idx, col_name in enumerate(target_cols):
                    row_dict[col_name] = float(row_vals[col_idx])
                submission_data.append(row_dict)

        # Save submission
        sub_df = pd.DataFrame(submission_data)
        sub_path = "./submission/submission.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
