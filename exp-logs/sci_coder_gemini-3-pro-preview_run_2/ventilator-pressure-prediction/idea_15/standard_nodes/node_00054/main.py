import sys
import os
import torch
import numpy as np
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, compute_metric
from library.dataset import prepare_data
from library.model import DFLB_BiLSTM
from library.train import Trainer, generate_submission

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Limit epochs to ensure execution finishes within 2 hours while maintaining performance.
# 12 epochs on A100 with the full dataset should take approx 30-40 minutes.
Config.EPOCHS = 12
Config.SCHEDULER_T_MAX = 12


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    # debug=False ensures we use the full dataset to achieve the target metric.
    # load_cached_data=True attempts to use pre-processed data if available.
    print("Preparing data...")
    train_loader, val_loader, test_loader = prepare_data(
        debug=False, load_cached_data=True
    )

    # Determine input dimension from a sample batch
    sample_x, _, _ = next(iter(train_loader))
    input_dim = sample_x.shape[-1]
    print(f"Input feature dimension: {input_dim}")

    # 3. Model Initialization
    model = DFLB_BiLSTM(input_dim=input_dim).to(device)

    # 4. Training
    print(f"Initializing trainer (Epochs: {Config.EPOCHS})...")
    trainer = Trainer(model, device)

    # Explicitly pass the overridden epochs
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # 5. Final Validation & Failure Analysis
    print("\nRunning final validation and failure analysis...")

    # Load the best model checkpoint
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model weights.")

    model.eval()

    val_preds = []
    val_targets = []
    val_u_out = []
    val_inputs = []

    # Inference on validation set (no gradients)
    with torch.no_grad():
        for x, y, u_out in val_loader:
            x = x.to(device)
            y = y.to(device)
            u_out = u_out.to(device)

            preds = model(x)

            # Store data on CPU for analysis
            val_preds.append(preds.cpu())
            val_targets.append(y.cpu())
            val_u_out.append(u_out.cpu())
            val_inputs.append(x.cpu())

    # Concatenate all batches
    preds_tensor = torch.cat(val_preds)
    targets_tensor = torch.cat(val_targets)
    u_out_tensor = torch.cat(val_u_out)
    inputs_tensor = torch.cat(val_inputs)

    # Compute Final Metric
    final_metric = compute_metric(preds_tensor, targets_tensor, u_out_tensor)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Features
    # Calculate absolute error
    errors = torch.abs(preds_tensor - targets_tensor)

    # Convert to numpy for correlation calculation
    # We analyze the inspiratory phase primarily as it's the scored metric,
    # but correlations are calculated on the full set masked or unmasked.
    # Let's use the inspiratory phase mask to be consistent with the metric.
    insp_mask = u_out_tensor == 0

    errors_np = errors[insp_mask].numpy()
    inputs_np = inputs_tensor[insp_mask].numpy()

    # Feature names corresponding to dataset.py logic
    feature_names = [
        "time_step",
        "u_in",
        "R",
        "C",
        "dt",
        "volume",
        "R_u_in",
        "vol_C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff1",
        "u_in_diff2",
        "u_out",
    ]

    print("\nFailure Analysis - Feature Correlation with Error (Inspiratory Phase):")
    for i, name in enumerate(feature_names):
        if i < inputs_np.shape[1]:
            feat_values = inputs_np[:, i]
            # Check for constant values to avoid warning/NaN
            if np.std(feat_values) > 1e-9:
                corr, _ = pearsonr(errors_np, feat_values)
                print(f"Feature '{name}': {corr:.5f}")
            else:
                print(f"Feature '{name}': Constant (No Correlation)")

    # 6. Submission
    # Threshold from task description
    THRESHOLD = 0.1619843989610672

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        predictions = trainer.predict(test_loader)
        generate_submission(predictions)
    else:
        print(
            f"\nMetric {final_metric} does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
