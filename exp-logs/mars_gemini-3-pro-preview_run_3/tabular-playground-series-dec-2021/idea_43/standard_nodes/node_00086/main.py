import sys
import os
import torch
import numpy as np
import pandas as pd

# Append current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.model import DualViewDCNResNet
from library.data_loader import get_dataloaders
from library.train import Trainer


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override epochs for a fast baseline execution that fits within time limits
    # 12 epochs on A100 with 2.8M rows is efficient and sufficient for convergence
    Config.EPOCHS = 12

    seed_everything(Config.SEED, deterministic=Config.DETERMINISTIC_CUDNN)
    device = get_device()
    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading data...")
    # load_cached_data=True speeds up the process by using pre-saved numpy arrays
    train_loader, val_loader, test_loader, input_dim = get_dataloaders(
        load_cached_data=True
    )

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print(f"Initializing model with input dim: {input_dim}")
    model = DualViewDCNResNet(input_dim=input_dim, num_classes=Config.NUM_CLASSES)
    model.to(device)

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------
    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit()

    # ---------------------------------------------------------
    # 5. Evaluation & Failure Analysis
    # ---------------------------------------------------------
    print("Loading best model for evaluation...")
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model file not found. Using current model state.")

    model.eval()

    print("Computing validation metrics and failure analysis...")
    val_preds = []
    val_targets = []
    val_features = []

    # Validation Inference
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            # Targets stay on CPU for metric calc to save GPU memory

            # Forward pass (primary head only)
            logits, _ = model(batch_X)
            preds = torch.argmax(logits, dim=1)

            val_preds.append(preds.cpu().numpy())
            val_targets.append(batch_y.numpy())
            # Store features for failure analysis
            val_features.append(batch_X.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_features = np.vstack(val_features)

    # Metric Calculation
    accuracy = (val_preds == val_targets).mean()
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {accuracy:.10f}")

    # Failure Analysis: Correlation
    print("Performing Failure Analysis...")
    errors = (val_preds != val_targets).astype(int)  # 1 = Error, 0 = Correct

    # Compute correlation between each feature and the error vector
    n_features = val_features.shape[1]
    correlations = []

    error_std = np.std(errors)
    if error_std > 1e-9:
        for i in range(n_features):
            feat = val_features[:, i]
            feat_std = np.std(feat)
            if feat_std > 1e-9:
                # Pearson correlation
                corr = np.corrcoef(feat, errors)[0, 1]
                correlations.append((i, corr))
            else:
                correlations.append((i, 0.0))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("\nTop 10 Features correlated with Error (Index, Correlation):")
        for idx, corr in correlations[:10]:
            print(f"Feature {idx}: {corr:.4f}")
    else:
        print("Error variance is near zero, skipping correlation analysis.")

    # ---------------------------------------------------------
    # 6. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.9626291666666666

    if accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy meets threshold ({THRESHOLD}). Generating submission..."
        )

        test_ids = []
        test_preds = []

        with torch.no_grad():
            for batch_X, batch_ids in test_loader:
                batch_X = batch_X.to(device)

                logits, _ = model(batch_X)
                preds = torch.argmax(logits, dim=1)

                test_preds.append(preds.cpu().numpy())
                test_ids.append(batch_ids.numpy())

        test_preds = np.concatenate(test_preds)
        test_ids = np.concatenate(test_ids)

        # Convert 0-based index back to 1-based class labels
        test_preds = test_preds + 1

        submission = pd.DataFrame({"Id": test_ids, "Cover_Type": test_preds})

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation accuracy ({accuracy:.6f}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
