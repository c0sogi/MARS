import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config, set_seed
from library.train import train_model
from library.predict import generate_submission
from library.data_loader import get_dataloaders


def main():
    # 1. Setup and Configuration
    # Ensure reproducibility
    set_seed(Config.SEED)

    # 2. Model Training
    # train_model() handles the entire training loop, including:
    # - Data loading (with caching)
    # - Optimizer/Scheduler setup
    # - Training/Validation per epoch
    # - Early stopping
    # - Saving the best model checkpoint
    # - Loading best weights into the model before returning
    print("Starting RHTN training pipeline...")
    model = train_model()

    # 3. Validation Assessment
    # We perform a dedicated inference pass on the validation set to calculate
    # the final metric and perform failure analysis.
    print("Running validation assessment...")

    device = torch.device(Config.DEVICE)
    model.eval()

    # Retrieve dataloaders (uses cache, so it's fast)
    _, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    val_probs = []
    val_targets = []
    val_metas = []

    # Inference loop (No Grad for efficiency)
    with torch.no_grad():
        for inputs, meta, labels in val_loader:
            inputs = inputs.to(device)
            meta = meta.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(inputs, meta)
            probs = torch.sigmoid(logits)

            # Collect results
            val_probs.extend(probs.cpu().numpy().flatten())
            val_targets.extend(labels.cpu().numpy().flatten())
            val_metas.extend(meta.cpu().numpy().flatten())

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)
    val_metas = np.array(val_metas)

    # Calculate Final Metric (Log Loss)
    # Using sklearn's log_loss for robustness
    final_metric = log_loss(val_targets, val_probs, labels=[0, 1])

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    # Calculate error magnitude per sample
    errors = np.abs(val_targets - val_probs)

    # Calculate correlation between Error and Incidence Angle
    # This helps identify if the model struggles with specific radar angles
    df_analysis = pd.DataFrame({"error": errors, "inc_angle": val_metas})

    # Pearson correlation
    corr_inc = df_analysis["error"].corr(df_analysis["inc_angle"])
    print(f"Correlation between Error and Incidence Angle: {corr_inc}")

    # 5. Submission Generation
    # Threshold condition as specified in the task
    THRESHOLD = 0.20320119103524176

    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} meets threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"Metric {final_metric} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
