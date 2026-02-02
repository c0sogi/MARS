import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.data import get_dataloaders
from library.model import HighCapacityCompositeModel
from library.train import Trainer
from library.utils import seed_everything, get_config_hash, compute_metric


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for a fast baseline execution
    Config.epochs = 6
    Config.debug = False  # Use full data to ensure metric reliability

    # Ensure reproducibility
    seed_everything(Config.seed)

    print(
        f"Configuration: Epochs={Config.epochs}, Debug={Config.debug}, Device={Config.device}"
    )

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n=== Data Loading ===")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.debug,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        load_cached_data=True,
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Training
    # --------------------------------------------------------------------------
    print("\n=== Model Initialization ===")
    model = HighCapacityCompositeModel(config=Config)

    print("\n=== Starting Training ===")
    trainer = Trainer(model, train_loader, val_loader, config=Config)
    trainer.fit()

    # --------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n=== Validation & Failure Analysis ===")

    # Load the best model saved during training
    print(f"Loading best model from {Config.model_path}...")
    model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
    model.to(Config.device)
    model.eval()

    val_preds = []
    val_targets = []
    val_uouts = []
    val_inputs = []

    # Inference loop on validation set
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(Config.device)
            y = batch["y"].to(Config.device)
            u_out = batch["u_out"].to(Config.device)

            # Forward pass (final_head only)
            pred, _ = model(x)

            val_preds.append(pred.cpu())
            val_targets.append(y.cpu())
            val_uouts.append(u_out.cpu())
            val_inputs.append(x.cpu())

    # Concatenate and flatten
    val_preds = torch.cat(val_preds).numpy().flatten()
    val_targets = torch.cat(val_targets).numpy().flatten()
    val_uouts = torch.cat(val_uouts).numpy().flatten()

    # Flatten inputs for correlation analysis: (N_breaths * 80, N_features)
    val_inputs = torch.cat(val_inputs).numpy()
    val_inputs = val_inputs.reshape(-1, val_inputs.shape[-1])

    # Compute Final Metric (Masked MAE)
    mask = val_uouts == 0
    final_mae = np.mean(np.abs(val_preds[mask] - val_targets[mask]))

    print(f"Final Validation Metric: {final_mae}")

    # Failure Analysis: Correlation of Error with Features
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    errors = np.abs(val_preds - val_targets)

    # Filter for inspiratory phase only
    errors_masked = errors[mask]
    inputs_masked = val_inputs[mask]

    for i, feat_name in enumerate(Config.features):
        feat_vals = inputs_masked[:, i]
        # Calculate correlation if variance exists
        if np.std(feat_vals) > 1e-9:
            corr = np.corrcoef(errors_masked, feat_vals)[0, 1]
            print(f"  {feat_name}: {corr:.4f}")
        else:
            print(f"  {feat_name}: 0.0000 (Constant)")

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    threshold = 0.2164510190486908

    if final_mae < threshold:
        print(f"\nMetric {final_mae} < {threshold}. Generating submission...")

        # Load test IDs from cache
        feature_hash = get_config_hash(Config.features)
        suffix = f"{feature_hash}_debug" if Config.debug else feature_hash
        test_ids_path = os.path.join(Config.cache_dir, f"test_ids_{suffix}.npy")

        if not os.path.exists(test_ids_path):
            print(f"Error: Test IDs file not found at {test_ids_path}")
            return

        test_ids = np.load(test_ids_path)
        test_preds = []

        # Inference loop on test set
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(Config.device)
                pred, _ = model(x)
                test_preds.append(pred.cpu())

        test_preds = torch.cat(test_preds).numpy().flatten()

        # Sanity check on lengths
        if len(test_preds) != len(test_ids):
            print(
                f"Warning: Prediction length ({len(test_preds)}) matches ID length ({len(test_ids)})?"
            )
            # Truncate to match if necessary (though pipeline ensures alignment)
            min_len = min(len(test_preds), len(test_ids))
            test_preds = test_preds[:min_len]
            test_ids = test_ids[:min_len]

        # Create submission DataFrame
        submission = pd.DataFrame({"id": test_ids, "pressure": test_preds})

        # Save
        os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)
        submission.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")

    else:
        print(f"\nMetric {final_mae} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
