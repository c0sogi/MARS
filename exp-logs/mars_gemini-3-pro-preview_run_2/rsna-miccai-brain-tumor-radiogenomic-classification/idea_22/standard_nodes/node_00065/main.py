import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library import config, data, model as lib_model, train as lib_train


def main():
    # 1. Setup and Configuration
    # Set seeds for reproducibility
    config.seed_everything(config.SEED)

    print("Initializing Fast Baseline Pipeline...")

    # 2. Training
    # We use 15 epochs which is sufficient for convergence on this small dataset
    # while keeping runtime short.
    print("Starting training phase...")
    best_auc_from_train = lib_train.run_training(
        num_epochs=15, load_cached_data=True, patience=5
    )

    # 3. Validation and Failure Analysis
    print("\nStarting Validation and Failure Analysis...")

    # Load the best model state
    device = config.DEVICE
    model = lib_model.AsymmetricEfficientNet().to(device)
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("Error: Best model file not found.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get DataLoaders (re-using library function which handles caching)
    _, val_loader, test_loader = data.get_data_loaders(load_cached_data=True)

    # Load Validation Metadata for analysis mapping
    df_val = pd.read_csv(config.VAL_METADATA)

    # Run Inference on Validation Set
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()

            val_probs.extend(probs.flatten())
            val_targets.extend(targets.numpy().flatten())

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    # Calculate and Print Final Metric
    final_metric = roc_auc_score(val_targets, val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate absolute error
    errors = np.abs(val_targets - val_probs)

    # Extract features for correlation analysis
    # We correlate error with the number of slices in each modality to see
    # if scan depth/volume affects performance.
    analysis_features = []

    # Iterate through validation dataframe to get file stats
    # Note: We assume the order in df_val matches the loader (which is true for non-shuffled val loader)
    for idx, row in df_val.iterrows():
        feat_row = {}

        # Get slice counts for each modality
        for mod in config.MODALITIES:
            path_col = f"path_{mod}"
            full_path = os.path.join(config.INPUT_DIR, row[path_col])
            try:
                # Fast way to get file count without reading images
                # We use os.scandir for better performance than listdir
                count = sum(1 for _ in os.scandir(full_path))
            except (FileNotFoundError, OSError):
                count = 0
            feat_row[f"{mod}_slices"] = count

        feat_row["error"] = errors[idx]
        analysis_features.append(feat_row)

    df_analysis = pd.DataFrame(analysis_features)

    # Calculate correlations
    print("\nFailure Analysis: Correlation between Error Magnitude and Input Features:")
    if not df_analysis.empty:
        # Compute correlation of 'error' with other columns
        correlations = df_analysis.corr()["error"].drop("error")
        print(correlations)
    else:
        print("No analysis data available.")

    # 4. Submission
    # Threshold defined in the task
    THRESHOLD = 0.6303636363636363

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        output_csv = "./submission/submission.csv"
        lib_model.predict_and_submit(model, test_loader, device, output_path=output_csv)
    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
