import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, save_submission
from library.data_loader import get_dataloaders
from library.model import AsymmetricParallelNet
from library.train import train_model, inference


def run():
    # =========================================================================
    # 1. Configuration & Initialization
    # =========================================================================
    # Override Config for Fast Baseline Execution
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 8192  # Increase batch size for A100 efficiency

    # Initialize environment
    Config.initialize()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # =========================================================================
    # 2. Data Loading & Subsampling
    # =========================================================================
    print("Loading data...")
    # Load cached data if available
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # Limit maximum number of training samples for fast baseline
    MAX_TRAIN_SAMPLES = 500000
    if len(train_loader.dataset) > MAX_TRAIN_SAMPLES:
        print(f"Subsampling training data to {MAX_TRAIN_SAMPLES} samples...")
        # Use a generator for reproducibility
        g = torch.Generator()
        g.manual_seed(Config.SEED)
        indices = torch.randperm(len(train_loader.dataset), generator=g)[
            :MAX_TRAIN_SAMPLES
        ]

        train_subset = Subset(train_loader.dataset, indices)
        train_loader_fast = DataLoader(
            train_subset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
    else:
        train_loader_fast = train_loader

    # =========================================================================
    # 3. Model Setup
    # =========================================================================
    # Detect input dimension from a batch
    sample_batch, _ = next(iter(train_loader_fast))
    input_dim = sample_batch.shape[1]
    print(f"Detected Input Dimension: {input_dim}")

    print("Initializing AsymmetricParallelNet...")
    model = AsymmetricParallelNet(input_dim=input_dim, num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # =========================================================================
    # 4. Training
    # =========================================================================
    # train_model returns the model with the best validation weights loaded
    model = train_model(model, train_loader_fast, val_loader, device)

    # =========================================================================
    # 5. Validation Assessment & Failure Analysis
    # =========================================================================
    print("Performing final validation assessment...")
    model.eval()

    val_preds_list = []
    val_targets_list = []
    val_inputs_list = []

    # Inference loop without gradients for speed
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Move to CPU to save GPU memory
            val_preds_list.append(preds.cpu().numpy())
            val_targets_list.append(labels.cpu().numpy())
            val_inputs_list.append(inputs.cpu().numpy())

    # Concatenate all batches
    val_preds = np.concatenate(val_preds_list)
    val_targets = np.concatenate(val_targets_list)
    val_inputs = np.concatenate(val_inputs_list)

    # Calculate Metric
    accuracy = (val_preds == val_targets).mean()
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis
    print("\nRunning Failure Analysis...")
    # Error magnitude: 1 if incorrect, 0 if correct
    errors = (val_preds != val_targets).astype(int)

    # Calculate correlation between each feature and the error
    n_features = val_inputs.shape[1]
    correlations = []

    # Pre-calculate std of errors to avoid recomputing
    std_errors = np.std(errors)

    if std_errors > 0:
        for i in range(n_features):
            feat_vals = val_inputs[:, i]
            std_feat = np.std(feat_vals)

            if std_feat > 0:
                # Pearson correlation
                corr = np.corrcoef(feat_vals, errors)[0, 1]
                correlations.append((i, corr))
            else:
                correlations.append((i, 0.0))
    else:
        print("No errors found (Perfect Accuracy). Skipping correlation analysis.")

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for idx, corr in correlations[:5]:
        print(f"  Feature Index {idx}: Correlation {corr:.6f}")

    # =========================================================================
    # 6. Submission
    # =========================================================================
    THRESHOLD = 0.9626291666666666

    if accuracy > THRESHOLD:
        print(f"\nValidation metric {accuracy} > {THRESHOLD}. Generating submission...")
        predictions = inference(model, test_loader, device)
        save_submission(predictions, test_ids, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric {accuracy} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    run()
