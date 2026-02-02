import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import SEED, DEVICE, EPOCHS
from library.utils import set_seed, load_checkpoint
from library.dataset import get_dataloaders
from library.trainer import Trainer


def main():
    # 1. Setup
    warnings.filterwarnings("ignore")
    set_seed(SEED)

    print("Initializing pipeline...")

    # 2. Data Loading
    # We use the full dataset to ensure we meet the high performance threshold.
    # The A100 GPU can handle this dataset size very quickly (minutes).
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        debug=False, load_cached_data=True
    )

    # 3. Training
    trainer = Trainer(train_loader, val_loader, test_loader, test_ids)
    trainer.fit(epochs=EPOCHS)

    # 4. Validation & Failure Analysis
    print("\nRunning Validation and Failure Analysis...")

    # Load the best model for evaluation
    if os.path.exists(trainer.best_model_path):
        load_checkpoint(trainer.model, None, trainer.best_model_path, device=DEVICE)
    else:
        print("Warning: No best model checkpoint found. Using current model state.")

    trainer.model.eval()

    val_probs = []
    val_targets = []

    # Features for failure analysis
    # We will track basic statistics of the input spectrograms
    feat_means = []
    feat_stds = []
    feat_maxs = []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            # Forward pass
            output = trainer.model(data)
            probs = torch.sigmoid(output).cpu().numpy().flatten()

            val_probs.extend(probs)
            val_targets.extend(target.cpu().numpy().flatten())

            # Extract features for failure analysis (on CPU to save GPU mem)
            # data shape: (B, 1, F, T)
            # Flatten spatial dims: (B, F*T)
            flat_data = data.view(data.size(0), -1).cpu()
            feat_means.extend(flat_data.mean(dim=1).numpy())
            feat_stds.extend(flat_data.std(dim=1).numpy())
            feat_maxs.extend(flat_data.max(dim=1).values.numpy())

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    # Compute Metric
    final_metric = roc_auc_score(val_targets, val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate error magnitude
    errors = np.abs(val_targets - val_probs)

    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "spec_mean": feat_means,
            "spec_std": feat_stds,
            "spec_max": feat_maxs,
        }
    )

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    correlations = df_analysis.corr()["error"].drop("error")
    print(correlations)

    # 5. Submission
    # Threshold defined in task
    THRESHOLD = 0.9934990421176494

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nValidation metric ({final_metric}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
