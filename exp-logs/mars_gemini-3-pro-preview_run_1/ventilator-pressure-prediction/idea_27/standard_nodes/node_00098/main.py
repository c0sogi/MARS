import pandas as pd
import numpy as np
import torch
import sys
import os

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.train import Trainer
from library.model import validate, predict_test
from library.dataset import VentilatorDataset
from torch.utils.data import DataLoader


def main():
    # 1. Configure for Fast Baseline Execution
    # We have an A100 GPU, so we can afford more epochs than a CPU run.
    # 15 epochs should be sufficient to reach a good convergence point
    # while easily fitting within the 24-minute limit.
    Config.epochs = 15
    Config.debug = False  # Use full dataset

    # Ensure reproducibility
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)

    print("=== Starting Runfile Execution ===")
    print(f"Configuration: Epochs={Config.epochs}, Debug={Config.debug}")

    # 2. Train Model
    trainer = Trainer()
    trainer.fit()

    # 3. Validation Assessment
    print("\n=== Validation Assessment ===")
    val_mae = validate(trainer.model, trainer.val_loader, trainer.device)
    # Print strictly in the requested format with full precision
    print(f"Final Validation Metric: {val_mae}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    trainer.model.eval()

    all_errors = []
    all_features = []

    # Feature map based on VentilatorDataset feature_cols order
    # 0: time_step, 1: u_in, 2: u_out, 3: R, 4: C, 5: volume
    feat_indices = {"time_step": 0, "u_in": 1, "R": 3, "C": 4, "volume": 5}

    with torch.no_grad():
        for x, u_out, y in trainer.val_loader:
            x = x.to(trainer.device)
            u_out = u_out.to(trainer.device)
            y = y.to(trainer.device)

            # Forward pass
            pred, _ = trainer.model(x)
            pred = pred.squeeze(-1)

            # Calculate absolute error
            error = torch.abs(pred - y)

            # Filter for inspiratory phase only (u_out == 0)
            # u_out is 1 for expiratory, 0 for inspiratory
            mask = (1 - u_out) > 0.5

            if mask.sum() > 0:
                # Extract valid errors
                valid_errors = error[mask].cpu().numpy()
                all_errors.append(valid_errors)

                # Extract valid features
                # x is (Batch, Seq, Feat) -> flatten and mask
                x_masked = x[mask]  # (N_valid, Feat)
                all_features.append(x_masked.cpu().numpy())

    if all_errors:
        all_errors = np.concatenate(all_errors)
        all_features = np.concatenate(all_features)

        print("Correlation between Absolute Error and Features (Inspiratory Phase):")
        for name, idx in feat_indices.items():
            feat_vals = all_features[:, idx]
            # Compute correlation
            if len(feat_vals) > 1 and np.std(feat_vals) > 1e-9:
                corr = np.corrcoef(all_errors, feat_vals)[0, 1]
                print(f"  {name}: {corr:.4f}")
            else:
                print(f"  {name}: N/A (Constant feature)")
    else:
        print("No inspiratory phase data found for failure analysis.")

    # 5. Submission Generation
    threshold = 0.2164510190486908

    if val_mae < threshold:
        print(f"\nValidation metric {val_mae} is better than threshold {threshold}.")
        print("Generating submission file...")

        # Load Test Data
        test_ds = VentilatorDataset(split="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Predict
        flat_preds = predict_test(trainer.model, test_loader, trainer.device)

        # Load ID mapping
        test_df = pd.read_csv(Config.test_file)

        # Ensure correct order: breath_id, time_step
        # This matches the order in FeatureEngineer._add_physics_features
        test_df = test_df.sort_values([Config.breath_id_col, "time_step"]).reset_index(
            drop=True
        )

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": test_df["id"], "pressure": flat_preds})

        # Save
        os.makedirs(os.path.dirname(Config.output_submission_path), exist_ok=True)
        submission.to_csv(Config.output_submission_path, index=False)
        print(f"Submission saved to {Config.output_submission_path}")

    else:
        print(
            f"\nValidation metric {val_mae} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
