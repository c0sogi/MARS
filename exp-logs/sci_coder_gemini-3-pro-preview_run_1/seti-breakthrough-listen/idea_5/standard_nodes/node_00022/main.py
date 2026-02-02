import os
import sys
import numpy as np
import pandas as pd
import torch

# Import library modules
from library import config, utils, data, model, train


def run_failure_analysis(model_path):
    """
    Loads the trained model, runs inference on the validation set,
    calculates the final metric, and performs failure analysis.
    """
    print("Running failure analysis on validation set...")

    device = torch.device(config.DEVICE)

    # Initialize model
    net = model.TimeDistributedResNet50GN()
    net = net.to(device)

    # Load checkpoint
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return 0.0

    epoch, score = utils.load_checkpoint(model_path, net, device=config.DEVICE)
    print(f"Loaded model from epoch {epoch} with AUC {score}")

    net.eval()

    # Get validation dataloader
    # We use the same batch size as training
    val_loader = data.get_val_dataloader(
        batch_size=config.BATCH_SIZE, debug=config.DEBUG
    )

    all_targets = []
    all_preds = []

    # Lists to store features for failure analysis
    # We will compute simple statistics on the input spectrograms
    feat_mean = []
    feat_std = []
    feat_max = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Inference
            outputs = net(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            # Store predictions and targets
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy().flatten())

            # Compute features for analysis (on CPU to avoid OOM)
            # inputs shape: (B, 6, 1, 273, 256)
            # We flatten the spatial/time dimensions for simple stats
            inputs_cpu = inputs.cpu().numpy()
            b = inputs_cpu.shape[0]
            flat = inputs_cpu.reshape(b, -1)

            feat_mean.extend(np.mean(flat, axis=1))
            feat_std.extend(np.std(flat, axis=1))
            feat_max.extend(np.max(flat, axis=1))

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # 1. Calculate and Print Final Metric
    final_metric = utils.get_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 2. Failure Analysis
    # Calculate error magnitude
    errors = np.abs(all_targets - all_preds)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "input_mean": feat_mean,
            "input_std": feat_std,
            "input_max": feat_max,
        }
    )

    # Calculate correlations
    correlations = df_analysis.corr()["error"].drop("error")

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    print(correlations)

    return final_metric


def main():
    # Set seeds for reproducibility
    utils.seed_everything(config.SEED)

    # Override Config for Fast Baseline
    # We use 5 epochs to ensure the code completes quickly while allowing the ResNet50
    # sufficient time to converge given the OneCycleLR schedule.
    # We use the full dataset (DEBUG=False) to meet the high AUC threshold.
    config.EPOCHS = 5
    config.DEBUG = False

    print(
        f"Configuration: Epochs={config.EPOCHS}, Debug={config.DEBUG}, Batch Size={config.BATCH_SIZE}"
    )

    # 1. Train the model
    # train_model returns the path to the best checkpoint
    best_model_path = train.train_model(debug=config.DEBUG)

    # 2. Validate and Analyze
    final_metric = run_failure_analysis(best_model_path)

    # 3. Generate Submission if Threshold Met
    # Threshold provided in task description
    THRESHOLD = 0.8275886022045841

    if final_metric > THRESHOLD:
        print(f"\nMetric {final_metric} > {THRESHOLD}. Generating submission...")
        train.generate_submission(best_model_path, debug=config.DEBUG)
    else:
        print(f"\nMetric {final_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
