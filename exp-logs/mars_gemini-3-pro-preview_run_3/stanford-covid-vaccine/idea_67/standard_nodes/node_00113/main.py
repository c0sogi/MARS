import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, MCRMSE
from library.data import get_dataloaders
from library.model import DADBiGRUModel
from library.train import train_one_epoch, validate, inference


def failure_analysis(model, loader, device, metadata_path):
    """
    Performs failure analysis by correlating per-sample errors with input features.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []
    all_inputs = []

    # Gather predictions, targets, and inputs
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["ids"]

            outputs = model(inputs, pair_indices, pair_masks)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)
            all_inputs.append(inputs.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_inputs = torch.cat(all_inputs, dim=0)  # Shape: (N, 107, 14)

    # Calculate MCRMSE per sample
    # Slice to scored length (68)
    y_true = all_targets[:, : Config.PRED_LEN, :]
    y_pred = all_preds[:, : Config.PRED_LEN, :]

    # Filter to scored columns: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    scored_indices = [0, 1, 3]
    y_true = y_true[:, :, scored_indices]
    y_pred = y_pred[:, :, scored_indices]

    # Compute RMSE per sample (averaging over length and columns)
    mse_per_sample = torch.mean((y_true - y_pred) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Extract Features from Inputs for Correlation Analysis
    # Input Channels: 0:A, 1:G, 2:C, 3:U, 4:(, 5:), 6:.
    seq_len = 107
    pct_A = all_inputs[:, :, 0].sum(dim=1).numpy() / seq_len
    pct_G = all_inputs[:, :, 1].sum(dim=1).numpy() / seq_len
    pct_C = all_inputs[:, :, 2].sum(dim=1).numpy() / seq_len
    pct_U = all_inputs[:, :, 3].sum(dim=1).numpy() / seq_len

    # Paired content: sum of '(' and ')'
    pct_paired = (all_inputs[:, :, 4] + all_inputs[:, :, 5]).sum(
        dim=1
    ).numpy() / seq_len

    # Load metadata for Signal to Noise ratio
    sn_values = np.zeros_like(rmse_per_sample)
    try:
        if os.path.exists(metadata_path):
            df_meta = pd.read_parquet(metadata_path)
            # Map SN based on ID
            id_to_sn = dict(zip(df_meta["id"], df_meta["signal_to_noise"]))
            sn_values = np.array([id_to_sn.get(i, 0.0) for i in all_ids])
    except Exception as e:
        pass  # Fail silently if metadata issue, SN will be 0

    # Calculate Correlations
    features = {
        "pct_A": pct_A,
        "pct_G": pct_G,
        "pct_C": pct_C,
        "pct_U": pct_U,
        "pct_paired": pct_paired,
        "signal_to_noise": sn_values,
    }

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    for name, feat in features.items():
        if np.std(feat) > 1e-9:
            # Pearson correlation
            corr = np.corrcoef(rmse_per_sample, feat)[0, 1]
            print(f"  {name}: {corr:.4f}")
        else:
            print(f"  {name}: N/A (No variance)")


def main():
    # 1. Configuration Override for Fast Baseline
    # Reducing epochs to 10 to ensure rapid execution within time limits
    Config.EPOCHS = 10

    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    # 2. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 3. Data Loading
    # Use cached data for speed
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 4. Model Initialization
    model = DADBiGRUModel().to(device)

    # 5. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    # Adjust scheduler for reduced epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )
    criterion = nn.MSELoss()

    # 6. Training Loop
    best_score = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.6f}"
        )

    # 7. Final Evaluation
    # Load the best model for final scoring and analysis
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Compute final metric on the full validation set
    final_val_score = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_score}")

    # 8. Failure Analysis
    failure_analysis(model, val_loader, device, Config.VAL_DATA_PATH)

    # 9. Conditional Submission
    THRESHOLD = 0.5884495377540588

    if final_val_score < THRESHOLD:
        print("Validation score meets threshold. Generating submission...")

        # Inference on Test Set
        test_preds, test_ids = inference(model, test_loader, device)

        # Format Predictions
        # test_preds shape: (N_samples, 107, 5)
        N, L, C = test_preds.shape
        flat_preds = test_preds.reshape(-1, C)

        # Create id_seqpos column
        id_seqpos_list = []
        for sample_id in test_ids:
            for i in range(L):
                id_seqpos_list.append(f"{sample_id}_{i}")

        submission_df = pd.DataFrame(
            flat_preds,
            columns=["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"],
        )
        submission_df.insert(0, "id_seqpos", id_seqpos_list)

        # Save Submission
        sub_path = "./submission/submission.csv"
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"Validation score {final_val_score} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
