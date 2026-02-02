import torch
import numpy as np
import pandas as pd
import sys
import os
from torch_geometric.nn import global_max_pool

# Import library modules
import library.config as config
import library.data as data
import library.model as model_lib
import library.train as train_lib
import library.utils as utils


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Fast baseline settings to ensure execution within time limits
    config.EPOCHS = 2
    config.DEBUG_SAMPLE_SIZE = 50000  # Limit dataset size for speed

    # Set seeds for reproducibility
    train_lib.set_seed(config.SEED)

    print("=== Configuration ===")
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Debug Sample Size: {config.DEBUG_SAMPLE_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n=== Initializing DataLoaders ===")
    # load_cached_data=True allows using pre-processed artifacts if available
    train_loader, val_loader, test_loader = data.get_dataloaders(load_cached_data=True)

    # ---------------------------------------------------------
    # 3. Model Training
    # ---------------------------------------------------------
    print("\n=== Starting Training ===")
    # train_model handles the training loop, validation monitoring, and saves the best model
    model = model_lib.train_model(train_loader, val_loader)

    # ---------------------------------------------------------
    # 4. Final Validation & Metric
    # ---------------------------------------------------------
    print("\n=== Final Validation ===")
    criterion = model_lib.CosineLoss()

    # Ensure model is in eval mode and on correct device
    model.eval()
    model.to(config.DEVICE)

    val_loss, val_metric = model_lib.validate(
        model, val_loader, criterion, config.DEVICE
    )

    # REQUIRED FORMAT: Print the full precision metric
    print(f"Final Validation Metric: {val_metric}")

    # ---------------------------------------------------------
    # 5. Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Failure Analysis ===")

    errors = []
    num_pulses = []
    max_charges = []

    # Iterate through validation set to collect stats
    # We disable gradients for efficiency
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(config.DEVICE)
            out = model(batch)

            # 1. Calculate Angular Errors per event
            # Normalize predictions to unit vectors
            pred_norm = torch.nn.functional.normalize(out, p=2, dim=1).cpu().numpy()
            target_np = batch.y.cpu().numpy()

            # Dot product -> Angle (clipping for numerical stability)
            dot_prod = np.sum(pred_norm * target_np, axis=1)
            dot_prod = np.clip(dot_prod, -1.0, 1.0)
            batch_errors = np.arccos(dot_prod)
            errors.extend(batch_errors)

            # 2. Extract Features (Pulse Count)
            # Use batch.ptr if available to get nodes per graph, else use batch index
            if hasattr(batch, "ptr"):
                # ptr is [0, n1, n1+n2, ...]
                counts = (batch.ptr[1:] - batch.ptr[:-1]).cpu().numpy()
            else:
                _, counts = torch.unique(batch.batch, return_counts=True)
                counts = counts.cpu().numpy()
            num_pulses.extend(counts)

            # 3. Extract Features (Max Charge)
            # batch.x column 4 is normalized charge. We need max per graph.
            charge_col = batch.x[:, 4:5]
            max_q = global_max_pool(charge_col, batch.batch).flatten().cpu().numpy()
            max_charges.extend(max_q)

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(
        {"error": errors, "num_pulses": num_pulses, "max_charge_norm": max_charges}
    )

    # Compute Correlations
    corr_pulses = df_analysis["error"].corr(df_analysis["num_pulses"])
    corr_charge = df_analysis["error"].corr(df_analysis["max_charge_norm"])

    print(f"Correlation (Error vs Num Pulses): {corr_pulses:.10f}")
    print(f"Correlation (Error vs Max Charge): {corr_charge:.10f}")
    print(
        "Interpretation: Negative correlation implies more data/charge leads to lower error."
    )

    # ---------------------------------------------------------
    # 6. Conditional Submission
    # ---------------------------------------------------------
    THRESHOLD = 1.5329564173900305

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric {val_metric} < {THRESHOLD}. Generating submission..."
        )
        model_lib.generate_submission(model, test_loader)
    else:
        print(f"\nValidation metric {val_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
