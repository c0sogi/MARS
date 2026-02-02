import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.train import Trainer
from library.data import get_dataloaders


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Adjust configuration for a fast but high-performance baseline
    # We increase epochs to 35 to capture the long-tail convergence at lower learning rates.
    # (Cite solution_lesson_node_00067)
    Config.EPOCHS = 35

    # We keep DEBUG=False to use the full dataset (640k samples).
    # On an A100 GPU, training on this volume is very fast, and using the full
    # data is crucial to meet the high AUC threshold (>0.997).
    Config.DEBUG = False

    # Ensure directories exist
    Config.setup()

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    print("Initializing Trainer...")
    trainer = Trainer()

    print("Starting Training...")
    # fit() trains the model, saves the best checkpoint, and returns the test loader
    test_loader = trainer.fit()

    # --------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Starting Validation & Failure Analysis ---")

    # Retrieve DataLoaders to access the validation set
    # (Trainer handles this internally, but we need explicit access here)
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )

    # Load the best model checkpoint
    device = torch.device(Config.DEVICE)
    model = trainer.model
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Error: Best model checkpoint not found.")
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Run Inference on Validation Set
    val_preds = []
    val_targets = []
    val_inputs_cont = []

    with torch.no_grad():
        for cat, cont, target in val_loader:
            cat = cat.to(device)
            cont = cont.to(device)
            target = target.to(device)

            # Forward pass
            outputs = model(cat, cont).squeeze()

            # Store results
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(target.cpu().numpy())
            val_inputs_cont.append(cont.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_inputs_cont = np.concatenate(val_inputs_cont)

    # Compute Final Metric
    final_auc = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation of Error with Continuous Features
    print("\nFailure Analysis: Correlation between Error and Continuous Features")
    errors = np.abs(val_targets - val_preds)

    # Reconstruct feature names (f_00 to f_30, excluding f_27)
    cont_feature_indices = [i for i in range(31) if i != 27]
    cont_feature_names = [f"f_{i:02d}" for i in cont_feature_indices]

    correlations = []
    # Calculate correlation for each feature
    for i in range(val_inputs_cont.shape[1]):
        feat_vals = val_inputs_cont[:, i]
        if np.std(feat_vals) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(errors, feat_vals)[0, 1]
        correlations.append((cont_feature_names[i], corr))

    # Sort by magnitude of correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # Print top 5 correlations
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 4. Submission
    # --------------------------------------------------------------------------
    threshold = 0.9972336610045187

    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )
        trainer.predict(test_loader)
    else:
        print(
            f"\nValidation metric ({final_auc}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
