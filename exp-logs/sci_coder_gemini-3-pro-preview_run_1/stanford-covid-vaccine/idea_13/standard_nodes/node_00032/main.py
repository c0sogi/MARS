import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import torch.optim as optim

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import RNAResidualBiGRU
from library.engine import fit, validate, generate_submission


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis by correlating prediction errors with input features.
    """
    print("\n=== Running Failure Analysis ===")
    model.eval()

    # 1. Get Predictions and Targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            seq_inputs = batch["seq_inputs"].to(device)
            pair_dists = batch["pair_dists"].to(device)
            loop_types = batch["loop_types"].to(device)
            targets = batch["targets"].to(device)

            preds = model(seq_inputs, pair_dists, loop_types)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)  # (N, Seq, Targets)
    all_targets = torch.cat(all_targets, dim=0)  # (N, Seq, Targets)

    # 2. Calculate MCRMSE per sample
    # Slice to scored positions
    preds_scored = all_preds[:, : Config.SEQ_SCORED, :]
    targets_scored = all_targets[:, : Config.SEQ_SCORED, :]

    # Squared error: (N, Scored, Targets)
    squared_error = (targets_scored - preds_scored) ** 2

    # MSE per target per sample: (N, Targets) -> Mean over seq dim (1)
    mse_per_sample_target = torch.mean(squared_error, dim=1)

    # RMSE per target per sample: (N, Targets)
    rmse_per_sample_target = torch.sqrt(mse_per_sample_target)

    # MCRMSE per sample: (N, ) -> Mean over target dim (1)
    sample_errors = torch.mean(rmse_per_sample_target, dim=1).numpy()

    # 3. Load Metadata for Features
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)

    # Ensure alignment: DataLoader should not shuffle validation data (shuffle=False in library)
    # But to be safe, we assume the order is preserved as per standard PyTorch Dataset behavior.

    # Extract Features
    features = {}

    # Metadata features
    if "signal_to_noise" in df_val.columns:
        features["signal_to_noise"] = df_val["signal_to_noise"].values
    if "SN_filter" in df_val.columns:
        features["SN_filter"] = df_val["SN_filter"].values

    # Sequence composition features
    sequences = df_val["sequence"].values
    features["len_A"] = np.array([s.count("A") for s in sequences])
    features["len_G"] = np.array([s.count("G") for s in sequences])
    features["len_C"] = np.array([s.count("C") for s in sequences])
    features["len_U"] = np.array([s.count("U") for s in sequences])

    # 4. Compute Correlations
    print(f"Analyzing {len(sample_errors)} validation samples...")
    print(f"{'Feature':<20} | {'Correlation':<12} | {'P-Value':<12}")
    print("-" * 50)

    for name, values in features.items():
        # Handle potential NaNs or type issues
        if len(values) != len(sample_errors):
            print(f"Skipping {name}: Length mismatch.")
            continue

        try:
            corr, p_val = pearsonr(values, sample_errors)
            print(f"{name:<20} | {corr:<12.4f} | {p_val:<12.4g}")
        except Exception as e:
            print(f"Could not calculate correlation for {name}: {e}")

    print("-" * 50)


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    # Cite debug_lesson_7: Regenerate cache to resolve schema mismatch (KeyError: 'seqs')
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # 3. Model Initialization
    print("Initializing Model...")
    model = RNAResidualBiGRU().to(device)

    # 4. Training Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # 5. Training Loop
    # fit() returns the model with the best weights loaded
    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.NUM_EPOCHS,
    )

    # 6. Final Validation
    print("Calculating final validation metric...")
    val_score = validate(model, val_loader, device)
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 8. Submission
    THRESHOLD = 0.6226052641868591

    if val_score < THRESHOLD:
        print(
            f"\nValidation score ({val_score:.6f}) meets threshold ({THRESHOLD:.6f}). Generating submission..."
        )

        # Ensure output directory exists
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        # Generate submission
        generate_submission(model, test_loader, device, submission_path)
    else:
        print(
            f"\nValidation score ({val_score:.6f}) did not meet threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
