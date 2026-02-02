import os
import pandas as pd
import numpy as np
import torch
import glob
from sklearn.metrics import roc_auc_score
from library import config, utils, data, model, train

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Reduce epochs to ensure execution within time limits while allowing convergence
config.EPOCHS = 5
# Ensure batch size is safe for the provided GPU
config.BATCH_SIZE = 32
# Set workers
config.NUM_WORKERS = 4


def get_validation_metadata_features(df):
    """
    Extracts simple metadata features (file counts) for failure analysis.
    """
    features = []
    # Iterate through validation dataframe
    for idx, row in df.iterrows():
        feat = {}
        for mod in config.MODALITIES:
            # Construct full path to modality folder
            rel_path = row[f"{mod.lower()}_path"]
            full_path = os.path.join(config.INPUT_DIR, rel_path)

            # Count files (proxy for brain volume/scan depth)
            try:
                # Fast check using glob
                count = len(glob.glob(os.path.join(full_path, "*.dcm")))
            except Exception:
                count = 0
            feat[f"{mod}_count"] = count
        features.append(feat)
    return pd.DataFrame(features)


def main():
    # Set global seed for reproducibility
    utils.seed_everything(config.SEED)

    print("Starting Fast Baseline Pipeline...")

    # ---------------------------------------------------------
    # 1. Training Phase
    # ---------------------------------------------------------
    # train.run_training handles the loop and saves the best model to 'working/best_model.pth'
    best_model_path = train.run_training(load_cached_data=True)

    # ---------------------------------------------------------
    # 2. Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\nRunning Validation Inference for Analysis...")
    device = utils.get_device()

    # Load the best model
    net = model.SIRVEfficientNet().to(device)
    checkpoint = utils.load_checkpoint(best_model_path, net)
    net.eval()

    # Get Validation Loader
    val_loader = data.get_dataloader("val", load_cached_data=True)

    all_preds = []
    all_targets = []
    all_ids = []

    # Inference Loop (No Gradient)
    with torch.no_grad():
        for images, targets, ids in val_loader:
            images = images.to(device)

            # Forward pass
            outputs = net(images)
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_preds.extend(preds)
            all_targets.extend(targets.numpy().flatten())
            all_ids.extend(ids.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_ids = np.array(all_ids)

    # Calculate Metric
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {auc}")

    # --- Failure Analysis ---
    print("\n=== Failure Analysis ===")

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)

    # Load Metadata to correlate errors with input features
    df_val = pd.read_csv(config.VAL_METADATA_PATH)

    # Extract file counts as proxy features for scan properties
    print("Extracting metadata features for correlation analysis...")
    meta_features_df = get_validation_metadata_features(df_val)

    # Create analysis dataframe
    # Note: We must align by BraTS21ID. The loader order depends on the cache/dataset order.
    # We create a dataframe from predictions and merge with metadata.
    pred_df = pd.DataFrame({"BraTS21ID": all_ids, "error": errors})

    # Merge predictions with metadata features
    analysis_df = pd.merge(
        pred_df, pd.concat([df_val, meta_features_df], axis=1), on="BraTS21ID"
    )

    # Calculate correlations
    feature_cols = [c for c in analysis_df.columns if "_count" in c]
    if feature_cols:
        correlations = (
            analysis_df[feature_cols + ["error"]].corr()["error"].drop("error")
        )
        print("Correlation between Model Error and Input Features (Slice Counts):")
        print(correlations)
    else:
        print("No numeric metadata features available for correlation.")

    # ---------------------------------------------------------
    # 3. Submission Generation
    # ---------------------------------------------------------
    threshold = 0.6705454545454544

    if auc > threshold:
        print(f"\nValidation AUC ({auc}) exceeds threshold ({threshold}).")
        print("Generating submission file...")
        train.predict_and_submit(best_model_path, load_cached_data=True)
    else:
        print(f"\nValidation AUC ({auc}) does not exceed threshold ({threshold}).")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
