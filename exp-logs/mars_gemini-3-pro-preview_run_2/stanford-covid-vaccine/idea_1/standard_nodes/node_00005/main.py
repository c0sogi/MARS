import os
import torch
import numpy as np
import pandas as pd
import sys

# Ensure library is in path
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.dataset import get_dataloaders
from library.model import RNAResNet
from library.trainer import Trainer


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)
    Config.create_dirs()

    # Hyperparameters
    EPOCHS = Config.EPOCHS
    BATCH_SIZE = Config.BATCH_SIZE

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RNAResNet(
        input_channels=Config.INPUT_CHANNELS, num_targets=Config.NUM_TARGETS
    ).to(device)

    # 4. Trainer Initialization
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=Config,
        device=device,
    )

    # 5. Training
    print("Starting Training...")
    trainer.fit(epochs=EPOCHS)

    # 6. Validation and Metric Calculation
    print("Computing Final Validation Metric...")
    model.eval()

    all_preds = []
    all_targets = []

    # Indices for scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    # deg_pH10(2) and deg_50C(4) are predicted but not scored in the metric.
    scored_indices = [0, 1, 3]

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)  # Shape: (N, 107, 5)
    all_targets = torch.cat(all_targets, dim=0)  # Shape: (N, 107, 5)

    # Filter predictions and targets to only the scored columns
    preds_scored_cols = all_preds[:, :, scored_indices]
    targets_scored_cols = all_targets[:, :, scored_indices]

    # Calculate MCRMSE using the provided loss class
    # The class handles slicing the sequence to Config.SCORED_LEN (68) internally
    criterion = MCRMSELoss()
    final_metric = criterion(preds_scored_cols, targets_scored_cols).item()

    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")

    # Calculate error per sample for correlation analysis
    # We manually slice to scored length for this analysis to match the metric logic
    scored_len = Config.SCORED_LEN
    p_trimmed = preds_scored_cols[:, :scored_len, :].numpy()
    t_trimmed = targets_scored_cols[:, :scored_len, :].numpy()

    # Calculate RMSE per sample (averaging over sequence length and the 3 scored targets)
    # Axis 1 = Sequence, Axis 2 = Targets
    mse_per_sample = np.mean((p_trimmed - t_trimmed) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load validation metadata to get features
    val_df = pd.read_csv(Config.VAL_CSV)

    if len(val_df) == len(rmse_per_sample):
        val_df["rmse_error"] = rmse_per_sample

        # Correlation with Signal to Noise
        if "signal_to_noise" in val_df.columns:
            sn_corr = val_df["signal_to_noise"].corr(val_df["rmse_error"])
            print(f"Correlation (Error vs Signal_to_Noise): {sn_corr}")

        # Correlation with Sequence Composition
        bases = ["A", "G", "C", "U"]
        print("Correlations with Nucleotide Composition:")
        for base in bases:
            # Calculate percentage of base in sequence
            val_df[f"pct_{base}"] = val_df["sequence"].apply(
                lambda x: x.count(base) / len(x)
            )
            corr = val_df[f"pct_{base}"].corr(val_df["rmse_error"])
            print(f"  Error vs %{base}: {corr}")
    else:
        print(
            "Warning: Validation dataset size mismatch. Skipping detailed correlation analysis."
        )

    # 8. Submission Generation
    THRESHOLD = 0.6795

    if final_metric < THRESHOLD:
        print(
            f"Validation Metric {final_metric} < {THRESHOLD}. Generating Submission..."
        )

        # Reload the best model saved by the trainer
        if os.path.exists(trainer.best_model_path):
            print(f"Loading best model from {trainer.best_model_path}")
            model.load_state_dict(
                torch.load(trainer.best_model_path, map_location=device)
            )
        else:
            print("Warning: Best model not found. Using current model weights.")

        model.eval()
        test_preds_list = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                test_preds_list.append(outputs.cpu().numpy())

        # Concatenate predictions: (N_test, 107, 5)
        test_predictions = np.concatenate(test_preds_list, axis=0)

        # Generate formatted submission file
        trainer.generate_submission(test_predictions)
    else:
        print(f"Validation Metric {final_metric} >= {THRESHOLD}. Skipping Submission.")

    print("Process Complete.")


if __name__ == "__main__":
    main()
