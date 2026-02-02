import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import provided library functions
from library.utils import seed_everything, get_device
from library.model import VAMSNet
from library.data import process_dataset, VAMSDataset
from library.train import train_model
from library.predict import generate_submission


def run():
    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Execution Device: {device}")

    # 2. Train the Model
    # Using 10 epochs as a fast baseline, relying on the library's internal caching
    print("\n--- Starting Training ---")
    best_model_path = train_model(
        epochs=10, batch_size=32, learning_rate=1e-4, load_cached_data=True
    )

    # 3. Validation Inference
    print("\n--- Performing Validation Inference ---")
    # Load validation metadata
    val_meta_path = "./metadata/val.parquet"
    val_df = pd.read_parquet(val_meta_path)

    # Process validation data (loads from cache created during training)
    X_val, y_val, ids_val = process_dataset(val_df, "val", load_cached_data=True)

    # Create DataLoader
    val_dataset = VAMSDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # Load Model
    model = VAMSNet(drop_path_rate=0.0).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Inference loop
    preds = []
    targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy().flatten())
            targets.extend(labels.numpy().flatten())

    preds = np.array(preds)
    targets = np.array(targets)

    # 4. Metrics
    auc = roc_auc_score(targets, preds)
    # STRICT OUTPUT FORMAT REQUIRED
    print(f"Final Validation Metric: {auc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(targets - preds)

    # Create a DataFrame for analysis to align features with errors
    # Note: process_dataset processes rows sequentially, so order is preserved.
    analysis_df = val_df.copy()
    analysis_df["error"] = errors

    # Extract simple meta-features (slice counts)
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    feature_cols = []

    for mod in modalities:
        col_name = f"{mod}_paths"
        feat_name = f"{mod}_count"
        # Calculate number of slices per modality
        analysis_df[feat_name] = analysis_df[col_name].apply(
            lambda x: len(x) if x is not None else 0
        )
        feature_cols.append(feat_name)

    print("Correlation between Error Magnitude and Input Features:")
    for feat in feature_cols:
        corr = analysis_df["error"].corr(analysis_df[feat])
        print(f" - {feat}: {corr:.6f}")

    # Check correlation with target class (is one class harder?)
    target_corr = analysis_df["error"].corr(analysis_df["MGMT_value"])
    print(f" - Target Class (MGMT_value): {target_corr:.6f}")

    # 6. Conditional Submission
    threshold = 0.6978181818181817
    print(f"\nMetric Check: {auc} > {threshold}?")

    if auc > threshold:
        print("Threshold met. Generating submission...")
        generate_submission(
            model_path=best_model_path, batch_size=32, load_cached_data=True
        )
    else:
        print("Threshold not met. Skipping submission generation.")


if __name__ == "__main__":
    run()
