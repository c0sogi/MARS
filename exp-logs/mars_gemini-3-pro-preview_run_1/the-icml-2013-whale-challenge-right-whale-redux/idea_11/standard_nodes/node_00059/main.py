import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.dataset import get_dataloaders
from library.model import HierarchicalCRNN
from library.train import Trainer, predict
from library.utils import calculate_auc


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # Load cached data for efficiency.
    # We use the full dataset to ensure we can meet the high AUC threshold.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = HierarchicalCRNN().to(device)

    # 4. Training
    # Initialize the Trainer with the model and loaders
    trainer = Trainer(model, train_loader, val_loader, device)
    # Execute training loop (uses Config.N_EPOCHS and Early Stopping)
    trainer.fit()

    # 5. Validation & Failure Analysis
    # Load the best model saved during training
    if os.path.exists(trainer.best_model_path):
        model.load_state_dict(torch.load(trainer.best_model_path, map_location=device))

    model.eval()
    val_preds = []
    val_targets = []
    val_stats = []

    # Run inference on validation set
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()

            val_preds.extend(probs)
            val_targets.extend(targets.numpy())

            # Collect input statistics for failure analysis
            # inputs shape: (B, 1, F, T)
            B = inputs.size(0)
            flat_inputs = inputs.view(B, -1)

            # Calculate mean and std of spectrogram intensity per sample
            means = flat_inputs.mean(dim=1).cpu().numpy()
            stds = flat_inputs.std(dim=1).cpu().numpy()

            for m, s in zip(means, stds):
                val_stats.append({"mean_intensity": m, "std_intensity": s})

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate and print the required metric
    final_auc = calculate_auc(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Perform Failure Analysis
    print("Performing Failure Analysis...")
    errors = np.abs(val_targets - val_preds)
    df_analysis = pd.DataFrame(val_stats)
    df_analysis["error"] = errors

    # Calculate correlations between error and input features
    for col in ["mean_intensity", "std_intensity"]:
        if col in df_analysis.columns:
            corr, _ = pearsonr(df_analysis[col], df_analysis["error"])
            print(f"Correlation between Error and {col}: {corr}")

    # 6. Submission
    threshold = 0.9946524988681537

    if final_auc > threshold:
        print(f"Validation metric {final_auc} > {threshold}. Generating submission...")

        # Generate predictions on test set
        test_preds = predict(model, test_loader, device)

        # Create submission DataFrame
        # test_loader.dataset.ids contains the filenames from metadata
        submission_df = pd.DataFrame(
            {"clip": test_loader.dataset.ids, "probability": test_preds}
        )

        # Save to disk
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"Validation metric {final_auc} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
