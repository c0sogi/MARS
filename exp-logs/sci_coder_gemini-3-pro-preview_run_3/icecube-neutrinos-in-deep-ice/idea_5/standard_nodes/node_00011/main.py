import os
import sys
import torch
import numpy as np
import warnings
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config, set_seed, setup_directories
from library.data_loader import get_dataloaders
from library.network import ADGN_Model, CosineSimilarityLoss
from library.training import train_one_epoch
from library.utils import spherical_to_cartesian, angular_dist_score
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def analyze_failures(preds_cart, targets_spherical, priors):
    """
    Computes correlations between error magnitude and event features.
    """
    # 1. Compute Angular Errors per event
    # Convert targets to Cartesian for dot product
    tx, ty, tz = spherical_to_cartesian(
        targets_spherical[:, 0], targets_spherical[:, 1]
    )

    # Dot product (Cosine similarity)
    # preds_cart is (N, 3), targets is (N, 3) implied by tx, ty, tz
    dot_prod = preds_cart[:, 0] * tx + preds_cart[:, 1] * ty + preds_cart[:, 2] * tz
    dot_prod = np.clip(dot_prod, -1.0, 1.0)
    errors = np.arccos(dot_prod)

    # 2. Extract Features from Priors
    # Index 16: Log10 Total Charge
    # Index 15: Duration
    log_charge = priors[:, 16]
    duration = priors[:, 15]

    # 3. Compute Correlations
    corr_charge, _ = pearsonr(errors, log_charge)
    corr_duration, _ = pearsonr(errors, duration)

    print(f"Correlation (Error vs LogCharge): {corr_charge}")
    print(f"Correlation (Error vs Duration): {corr_duration}")


def run():
    # 1. Setup
    setup_directories()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Configure for Fast Baseline
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 1024
    # We use manual limits via get_dataloaders, so keep DEBUG=False to avoid internal clipping
    Config.DEBUG = False

    # 2. Data Loading
    print("Loading DataLoaders...")
    # Limit training to ~100k events (100 batches) and validation to ~200k events (200 batches)
    # This ensures the run completes within 2 hours while providing significant stats.
    train_loader, val_loader = get_dataloaders(
        limit_train_batches=100, limit_val_batches=200
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = ADGN_Model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = CosineSimilarityLoss()

    # 4. Training
    print("Starting Training...")
    # Train for 1 epoch
    loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss: {loss}")

    # Save Model
    torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
    print("Model saved.")

    # 5. Validation & Failure Analysis
    print("Starting Validation and Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_priors = []

    with torch.no_grad():
        for X, priors, y, _ in val_loader:
            X = X.to(device, non_blocking=True)
            priors_dev = priors.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Inference
            preds = model(X, priors_dev)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_priors.append(
                priors.numpy()
            )  # priors was CPU tensor originally in loader, but moved to dev. Use original or cpu()

    # Concatenate results
    preds_cart = np.concatenate(all_preds, axis=0)
    targets_spherical = np.concatenate(all_targets, axis=0)
    priors_arr = np.concatenate(all_priors, axis=0)

    # Compute Metric
    # Convert Cartesian preds to Spherical for metric function
    p_az, p_zen = spherical_to_cartesian(
        preds_cart[:, 0], preds_cart[:, 1]
    )  # Wait, this is wrong function usage
    # We need cartesian_to_spherical for preds
    from library.utils import cartesian_to_spherical

    p_az, p_zen = cartesian_to_spherical(
        preds_cart[:, 0], preds_cart[:, 1], preds_cart[:, 2]
    )
    preds_spherical = np.stack([p_az, p_zen], axis=1)

    metric = angular_dist_score(targets_spherical, preds_spherical)
    print(f"Final Validation Metric: {metric}")

    # Run Failure Analysis
    analyze_failures(preds_cart, targets_spherical, priors_arr)

    # 6. Submission
    threshold = 1.5013689469017657
    if metric < threshold:
        print(
            f"Metric {metric} is better than threshold {threshold}. Generating submission..."
        )
        # Generate submission for the entire test set
        generate_submission(limit_batches=None)
    else:
        print(
            f"Metric {metric} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    run()
