import torch
import numpy as np
import pandas as pd
from library.config import Config, set_seed
from library.trainer import run_training, predict_and_submit
from library.dataset import get_dataloaders
from library.utils import calculate_auc


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Train the model
    # We use the full dataset (debug=False) and Config.EPOCHS to ensure we meet the
    # high performance threshold. The A100 GPU ensures this completes quickly.
    print("Starting training pipeline...")
    model, test_loader = run_training(
        epochs=Config.EPOCHS, load_cached_data=False, debug=False
    )

    # 3. Validation Inference for Metric & Failure Analysis
    print("Running validation inference...")
    # Retrieve val_loader. Data is already cached by run_training, so this is fast.
    _, val_loader, _ = get_dataloaders(load_cached_data=True, debug=False)

    model.eval()
    all_preds = []
    all_targets = []

    # Lists to store features for failure analysis
    feat_means = []
    feat_stds = []
    feat_maxs = []
    feat_ranges = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = labels.numpy()

            all_preds.extend(probs.flatten())
            all_targets.extend(targets)

            # Extract basic signal features from spectrograms for analysis
            # inputs shape: (B, 1, F, T)
            # Flatten to (B, F*T) to compute stats per sample
            flat_inputs = inputs.view(inputs.size(0), -1)
            flat_inputs_np = flat_inputs.cpu().numpy()

            feat_means.extend(flat_inputs_np.mean(axis=1))
            feat_stds.extend(flat_inputs_np.std(axis=1))
            batch_max = flat_inputs_np.max(axis=1)
            batch_min = flat_inputs_np.min(axis=1)
            feat_maxs.extend(batch_max)
            feat_ranges.extend(batch_max - batch_min)

    # 4. Calculate and Print Metric
    val_auc = calculate_auc(all_targets, all_preds)
    # REQUIRED FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    df_analysis = pd.DataFrame(
        {
            "target": all_targets,
            "pred": all_preds,
            "spec_mean": feat_means,
            "spec_std": feat_stds,
            "spec_max": feat_maxs,
            "spec_range": feat_ranges,
        }
    )

    # Calculate Error Magnitude
    df_analysis["error"] = np.abs(df_analysis["target"] - df_analysis["pred"])

    # Correlation
    feature_cols = ["spec_mean", "spec_std", "spec_max", "spec_range"]
    correlations = df_analysis[feature_cols + ["error"]].corr()["error"].drop("error")

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 6. Submission Logic
    # Threshold defined in task
    THRESHOLD = 0.9918798805807587

    if val_auc > THRESHOLD:
        print(
            f"\nMetric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, test_loader, device)
    else:
        print(
            f"\nMetric ({val_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
