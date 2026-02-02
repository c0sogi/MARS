import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import library modules
from library.config import Config
from library.data_utils import get_data
from library.dataset import ManufacturingDataset
from library.model import SRPFEModel
from library.engine import train_model, predict


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # Setup directories
    Config.setup()

    # 1. Configuration
    # Using full 50 epochs as per Lesson 00080 for maximum convergence
    print(f"Running pipeline with {Config.EPOCHS} epochs on device: {Config.DEVICE}")
    set_seed(Config.SEED)

    # 2. Data Loading
    # Load data using the library function (handles caching automatically)
    print("Loading data...")
    data_dict = get_data(load_cached_data=True)

    train_data = data_dict["train"]
    val_data = data_dict["val"]
    test_data = data_dict["test"]
    meta = data_dict["meta"]

    # Create Datasets
    # We use the full datasets as the A100 GPU can handle the throughput easily.
    train_ds = ManufacturingDataset(
        train_data["X_cat"], train_data["X_cont"], train_data["y"]
    )
    val_ds = ManufacturingDataset(val_data["X_cat"], val_data["X_cont"], val_data["y"])
    test_ds = ManufacturingDataset(test_data["X_cat"], test_data["X_cont"], y=None)

    # Create DataLoaders
    # Pin memory for faster host-to-device transfer
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing SR-PFE Model...")
    vocab_sizes = meta["vocab_sizes"]
    num_cont = len(meta["cont_cols"])

    model = SRPFEModel(vocab_sizes=vocab_sizes, num_cont_features=num_cont)

    # 4. Training
    # The engine handles the training loop, validation per epoch, and saving the best model.
    print("Starting training...")
    _ = train_model(model, train_loader, val_loader)

    # 5. Full Validation & Failure Analysis
    print("\nPerforming final validation inference for analysis...")

    # Load the best model saved during training
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        )
    else:
        print("Warning: Best model not found. Using current weights.")

    model.to(Config.DEVICE)
    model.eval()

    val_preds = []
    val_targets = []

    # Inference loop without gradient calculation
    with torch.no_grad():
        for batch in val_loader:
            x_cat = batch["cat"].to(Config.DEVICE)
            x_cont = batch["cont"].to(Config.DEVICE)
            y = batch["target"].to(Config.DEVICE)

            # Forward pass
            outputs = model(x_cat, x_cont)

            # Average probabilities across streams
            probs = torch.sigmoid(outputs)
            avg_probs = torch.mean(probs, dim=1)

            val_preds.append(avg_probs.cpu().numpy())
            val_targets.append(y.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets).flatten()

    # Calculate Final Metric
    final_metric = roc_auc_score(val_targets, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(val_targets - val_preds)

    # Calculate correlation between error and continuous features
    X_cont_val = val_data["X_cont"]
    cont_cols = meta["cont_cols"]

    correlations = []
    # Iterate through continuous columns to find correlation with error
    for i, col_name in enumerate(cont_cols):
        feat_vals = X_cont_val[:, i]
        # Correlation coefficient matrix [0,1] is the correlation between x and y
        corr = np.corrcoef(feat_vals, errors)[0, 1]
        correlations.append((col_name, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top features correlated with prediction error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 6. Submission Logic
    THRESHOLD = 0.9975746465492954

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric:.6f}) exceeds threshold ({THRESHOLD:.6f})."
        )
        print("Generating submission...")

        # Generate test predictions
        test_preds = predict(model, test_loader)

        # Create submission file
        submission = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        submission["target"] = test_preds

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nValidation metric ({final_metric:.6f}) does not exceed threshold ({THRESHOLD:.6f})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
