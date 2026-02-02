import sys
import os
import importlib
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
import library.data

importlib.reload(library.data)
from library.data import get_data
from library.engine import Engine, evaluate_model


def main():
    # 1. Setup and Configuration
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Adjust epochs for a faster baseline run while ensuring convergence
    Config.EPOCHS = 50

    # 2. Data Loading
    print("Initializing data pipeline...")
    # get_data handles feature caching/loading internally
    train_ds, val_ds, test_ds = get_data(load_cached_data=False)

    # Create DataLoaders
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

    # 3. Training
    # Initialize Engine and run the two-phase training
    engine = Engine(device=Config.DEVICE)
    model = engine.run_training(train_loader, val_loader)

    # 4. Final Validation Assessment
    print("\n=== Final Validation Assessment ===")
    final_metric = evaluate_model(model, val_loader, Config.DEVICE, metric="laplace")
    # Print exact format required
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    model.eval()

    errors = []
    features_list = []

    # Collect predictions and features from validation set
    with torch.no_grad():
        for batch in val_loader:
            imgs, tabs, weeks, targets = [x.to(Config.DEVICE) for x in batch]

            # Forward pass
            fvc_pred, _ = model(imgs, tabs, weeks)

            # Calculate absolute error
            batch_errors = torch.abs(targets - fvc_pred.squeeze()).cpu().numpy()
            errors.extend(batch_errors)

            # Extract features for correlation
            # tabs columns: [Age, Baseline_FVC, Baseline_Percent, ...] (normalized)
            # weeks: relative weeks
            # We concatenate Weeks and the first 3 continuous columns of tabs
            batch_feats = (
                torch.cat([weeks.view(-1, 1), tabs[:, :3]], dim=1).cpu().numpy()
            )
            features_list.extend(batch_feats)

    # Create DataFrame for analysis
    errors = np.array(errors)
    features_arr = np.array(features_list)
    feature_names = ["Weeks", "Age", "Baseline_FVC", "Baseline_Percent"]

    df_analysis = pd.DataFrame(features_arr, columns=feature_names)
    df_analysis["AbsError"] = errors

    # Calculate correlations
    correlations = df_analysis.corr()["AbsError"].drop("AbsError")
    print("Correlation between Model Error (MAE) and Input Features:")
    print(correlations)

    # 6. Submission Generation
    THRESHOLD = -7.004077599888947

    if final_metric > THRESHOLD:
        print(f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}).")
        engine.generate_submission(model, test_loader)
    else:
        print(
            f"\nValidation metric ({final_metric}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
