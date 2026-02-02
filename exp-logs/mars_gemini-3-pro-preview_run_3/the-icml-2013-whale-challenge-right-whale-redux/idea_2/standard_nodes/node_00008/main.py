import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.trainer import Trainer
from library.utils import set_seed


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # 1. Configuration for Fast Baseline
    # We set epochs to 15 to ensure the model converges sufficiently to beat the
    # high AUC threshold while still running quickly on the A100 GPU.
    Config.EPOCHS = 15

    # Set random seed for reproducibility
    set_seed(Config.SEED)

    # 2. Initialize Trainer
    # debug=False ensures we use the full dataset, which is necessary to achieve high performance.
    # The A100 GPU can handle the full dataset (approx 18k samples) very quickly.
    trainer = Trainer(debug=False)

    # 3. Train the model
    # This will run for Config.EPOCHS and save the best model based on validation AUC.
    trainer.fit()

    # 4. Validation & Failure Analysis
    print("Starting validation and failure analysis...")
    trainer.model.eval()
    val_loader = trainer.val_loader
    device = trainer.device

    all_preds = []
    all_targets = []

    # We will collect basic statistics of the input spectrograms to correlate with error
    feat_means = []
    feat_stds = []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)

            # Forward pass
            output = trainer.model(data)
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            targets_np = target.cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets_np)

            # Feature Extraction for Failure Analysis
            # data shape: [Batch, Channels, Freq, Time]
            # We compute the mean and std of the spectrogram values for each sample
            flat_inputs = data.view(data.size(0), -1)
            batch_means = flat_inputs.mean(dim=1).cpu().numpy()
            batch_stds = flat_inputs.std(dim=1).cpu().numpy()

            feat_means.extend(batch_means)
            feat_stds.extend(batch_stds)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Final Validation Metric
    final_auc = roc_auc_score(all_targets, all_preds)
    # Print metric in the exact required format
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    # Calculate absolute error magnitude
    errors = np.abs(all_targets - all_preds)

    # Create DataFrame to analyze correlations
    df_analysis = pd.DataFrame(
        {"error": errors, "input_mean": feat_means, "input_std": feat_stds}
    )

    # Calculate and print correlations
    correlations = df_analysis.corr()["error"].drop("error")
    print("\nCorrelation between Error Magnitude and Input Features:")
    print(correlations)

    # 5. Submission Generation
    # Threshold defined in the task description
    THRESHOLD = 0.9913801393656689

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        # The trainer's predict method handles loading the best model and saving to CSV
        trainer.predict()
    else:
        print(
            f"\nValidation metric ({final_auc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
