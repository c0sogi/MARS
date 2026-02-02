import os
import pandas as pd
import numpy as np
import torch
import scipy.stats as stats
from library.config import Config
from library.data import get_dataloaders
from library.engine import Engine
from library.utils import set_seed


def main():
    # 1. Configuration
    # Set output directory and ensure fast execution
    output_dir = "./working/idea_40"
    config = Config(output_dir=output_dir)

    # Fast baseline settings
    config.epochs = 15  # Limit epochs for speed
    config.batch_size = 32

    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    print("Initializing TAF-RDN Baseline...")
    print(f"Device: {config.device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # 3. Training
    engine = Engine(config)
    engine.train(train_loader, val_loader)

    # 4. Final Evaluation
    print("Loading best model for evaluation...")
    if os.path.exists(config.best_model_path):
        engine.model.load_state_dict(
            torch.load(config.best_model_path, map_location=config.device)
        )

    val_score = engine.evaluate(val_loader)
    print(f"Final Validation Metric: {val_score}")

    # 5. Failure Analysis
    print("\nRunning Failure Analysis on Validation Set...")
    engine.model.eval()

    # Collect predictions and targets
    all_preds = []
    all_targets = []

    # Scored indices for analysis: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    with torch.no_grad():
        for inputs, partner_indices, targets in val_loader:
            inputs = inputs.to(config.device)
            partner_indices = partner_indices.to(config.device)

            # Forward pass
            outputs = engine.model(inputs, partner_indices)
            final_pred = outputs[-1]  # (B, SeqLen, 5)

            # Slice to scored length and indices
            pred_sliced = final_pred[:, : config.pred_len, scored_indices].cpu().numpy()
            target_sliced = targets[:, : config.pred_len, scored_indices].numpy()

            all_preds.append(pred_sliced)
            all_targets.append(target_sliced)

    preds_arr = np.concatenate(all_preds, axis=0)  # (N, 68, 3)
    targets_arr = np.concatenate(all_targets, axis=0)  # (N, 68, 3)

    # Calculate Mean Squared Error per sample (averaged over positions and channels)
    # Shape: (N,)
    sample_mse = np.mean((preds_arr - targets_arr) ** 2, axis=(1, 2))
    sample_rmse = np.sqrt(sample_mse)

    # Load Metadata for correlation
    val_df = pd.read_csv(config.val_csv)

    # Ensure alignment (DataLoader should preserve order if shuffle=False)
    if len(val_df) != len(sample_rmse):
        print(
            f"Warning: Metadata length ({len(val_df)}) matches predictions ({len(sample_rmse)})?"
        )

    # Extract features
    sn_ratio = val_df["signal_to_noise"].values
    mean_reactivity = val_df["mean_reactivity"].values

    # Calculate Correlations
    corr_sn, _ = stats.pearsonr(sample_rmse, sn_ratio)
    corr_react, _ = stats.pearsonr(sample_rmse, mean_reactivity)

    print("-" * 40)
    print(f"Correlation (Error vs Signal-to-Noise): {corr_sn:.4f}")
    print(f"Correlation (Error vs Mean Reactivity): {corr_react:.4f}")
    print("-" * 40)

    if corr_sn < -0.1:
        print(
            "Observation: Higher signal-to-noise correlates with lower error (expected)."
        )
    elif corr_sn > 0.1:
        print(
            "Observation: Higher signal-to-noise correlates with higher error (unexpected)."
        )
    else:
        print("Observation: No strong linear correlation with signal-to-noise.")

    # 6. Submission
    threshold = 0.47142532743789534
    if val_score < threshold:
        print(
            f"\nValidation score {val_score} meets threshold ({threshold}). Generating submission..."
        )

        # Update config path to required submission location
        config.submission_path = "./submission/submission.csv"

        # Generate predictions
        engine.predict(test_loader)
    else:
        print(
            f"\nValidation score {val_score} does not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
