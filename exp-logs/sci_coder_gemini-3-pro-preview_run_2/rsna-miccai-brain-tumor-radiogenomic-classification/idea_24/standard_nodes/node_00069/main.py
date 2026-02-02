import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Import provided library modules
import library.config as config
from library import utils, data, model, train, inference


def get_slice_counts(df):
    """
    Helper to extract slice counts for failure analysis.
    """
    counts = {"flair": [], "t1w": [], "t1wce": [], "t2w": []}

    for _, row in df.iterrows():
        for mod in ["FLAIR", "T1w", "T1wCE", "T2w"]:
            path_col = f"path_{mod}"
            full_path = os.path.join(config.INPUT_DIR, row[path_col])
            count = 0
            if os.path.exists(full_path):
                try:
                    count = len(os.listdir(full_path))
                except OSError:
                    pass
            counts[mod.lower()].append(count)

    return pd.DataFrame(counts)


def main():
    # 1. Setup
    utils.seed_everything()
    device = utils.get_device()
    print(f"Execution Device: {device}")

    # 2. Train Model
    # Limiting to 10 epochs for a fast baseline execution as per requirements.
    print("\n--- Starting Training Phase ---")
    train.run_training(load_cached_data=True, max_epochs=10)

    # 3. Validation & Failure Analysis
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load Validation Data
    val_csv_path = os.path.join(config.METADATA_DIR, "val.csv")
    val_df = pd.read_csv(val_csv_path)

    # Get DataLoader (only val_loader is needed)
    _, val_loader, _ = data.get_dataloaders(
        train_df=None, val_df=val_df, test_df=None, load_cached_data=True
    )

    # Load Best Model
    net = model.AsymmetricEfficientNet(model_name=config.MODEL_NAME, pretrained=False)
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print("Error: Best model not found. Training may have failed.")
        return

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # Inference on Validation Set
    all_targets = []
    all_preds = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = net(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy().flatten())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # Compute Metric
    try:
        final_metric = roc_auc_score(all_targets, all_preds)
    except ValueError:
        final_metric = 0.5

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    errors = np.abs(all_targets - all_preds)

    # Extract features for correlation
    # Note: val_loader is not shuffled (shuffle=False), so order matches val_df
    meta_features = get_slice_counts(val_df)
    meta_features["error"] = errors

    # Calculate correlations
    correlations = meta_features.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 4. Submission
    threshold = 0.6303636363636363

    if final_metric > threshold:
        print(
            f"\nValidation Metric ({final_metric}) > Threshold ({threshold}). Generating Submission..."
        )
        inference.run_inference(load_cached_data=True)
    else:
        print(
            f"\nValidation Metric ({final_metric}) <= Threshold ({threshold}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
