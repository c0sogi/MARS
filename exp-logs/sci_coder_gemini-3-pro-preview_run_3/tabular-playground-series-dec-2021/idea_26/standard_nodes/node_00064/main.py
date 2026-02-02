import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Add the current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import DeepPreActDCNResNet
from library.train import Trainer

# Filter warnings for clean output
warnings.filterwarnings("ignore")


def run_pipeline():
    # 1. Setup & Configuration
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    # We load the full dataset. The A100 GPU allows processing 2.8M rows efficiently.
    # Using cached data if available to save preprocessing time.
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    # Config.INPUT_DIM is populated during get_dataloaders
    print("Initializing Deep Pre-Activation Parallel DCN-ResNet...")
    model = DeepPreActDCNResNet()

    # 4. Training
    # Initialize Trainer
    trainer = Trainer(model, device=device)

    # Fit the model
    # We use the default Config.EPOCHS (60). With batch size 4096 on A100,
    # training is expected to be well within the time limit.
    trainer.fit(train_loader, val_loader)

    # 5. Validation Assessment
    print("Performing final validation...")
    val_loss, val_acc = trainer.validate(val_loader)

    # REQUIRED: Print the final validation metric with full precision
    print(f"Final Validation Metric: {val_acc}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate correlation between prediction error and input features

    # Access validation features and targets directly from the dataset
    X_val = val_loader.dataset.X
    y_val = val_loader.dataset.y

    # Generate predictions for the validation set using the trained model
    model.eval()
    all_preds = []

    with torch.no_grad():
        # Iterate through loader to ensure correct batching/device movement
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu())

    all_preds = torch.cat(all_preds)

    # Calculate Error Vector (1 if prediction is wrong, 0 if correct)
    errors = (all_preds != y_val).float().numpy()
    X_val_np = X_val.numpy()

    # Calculate Pearson Correlation for each feature
    correlations = []
    num_features = X_val_np.shape[1]

    for i in range(num_features):
        feature_values = X_val_np[:, i]
        # Skip constant features to avoid division by zero in correlation
        if np.std(feature_values) == 0:
            continue

        corr, _ = pearsonr(feature_values, errors)
        correlations.append((i, corr))

    # Sort by absolute correlation magnitude (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with error (Feature Index: Correlation):")
    for idx, corr in correlations[:5]:
        print(f"Feature {idx}: {corr:.6f}")

    # 7. Submission Generation
    # Strict threshold as per task requirements
    THRESHOLD = 0.9625041666666667

    if val_acc > THRESHOLD:
        print(
            f"\nValidation metric {val_acc} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        trainer.predict(test_loader, test_ids)
    else:
        print(
            f"\nValidation metric {val_acc} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
