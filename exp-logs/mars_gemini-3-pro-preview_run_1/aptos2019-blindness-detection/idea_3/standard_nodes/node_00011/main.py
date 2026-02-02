import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from scipy.stats import pearsonr

# Import provided library modules
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model_lib
import library.engine as engine


def get_validation_predictions(model, loader, device):
    """
    Runs inference on the validation set to get raw predictions and targets
    for metric calculation and failure analysis.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].numpy()
            ids = batch["id_code"]

            logits = model(images)
            probs = torch.sigmoid(logits)
            scores = probs.sum(dim=1)
            preds = scores.round().cpu().numpy().astype(int)

            all_preds.extend(preds)
            all_targets.extend(targets)
            all_ids.extend(ids)

    return np.array(all_preds), np.array(all_targets), all_ids


def perform_failure_analysis(val_meta_path, ids, preds, targets):
    """
    Analyzes the correlation between model error and image meta-features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate Error Magnitude
    errors = np.abs(preds - targets)

    # Load metadata to get file paths
    df_meta = pd.read_csv(val_meta_path)
    # Ensure alignment by setting index to id_code
    df_meta = df_meta.set_index("id_code")
    # Reindex based on the order of ids from the dataloader
    df_meta = df_meta.loc[ids]

    # Feature accumulators
    widths = []
    heights = []
    aspect_ratios = []
    mean_intensities = []
    file_sizes = []

    # Iterate and extract features
    # Note: We read from input_dir based on df_meta['file_path']
    for _, row in df_meta.iterrows():
        full_path = os.path.join(config.INPUT_DIR, row["file_path"])

        try:
            # File size
            f_size = os.path.getsize(full_path)

            # Image stats
            img = cv2.imread(full_path)
            if img is None:
                # Fallback for missing images
                widths.append(0)
                heights.append(0)
                aspect_ratios.append(0)
                mean_intensities.append(0)
                file_sizes.append(0)
                continue

            h, w, c = img.shape
            mean_val = img.mean()

            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            mean_intensities.append(mean_val)
            file_sizes.append(f_size)

        except Exception:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            mean_intensities.append(0)
            file_sizes.append(0)

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
            "mean_intensity": mean_intensities,
            "file_size": file_sizes,
        }
    )

    # Calculate Correlations
    features = ["width", "height", "aspect_ratio", "mean_intensity", "file_size"]
    print("Correlation between Error Magnitude and Input Features:")
    for feat in features:
        if analysis_df[feat].std() == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(analysis_df["error"], analysis_df[feat])
        print(f"{feat}: {corr:.4f}")


def main():
    # 1. Setup
    utils.seed_everything()
    device = config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = dataset.create_dataloaders(
        config.TRAIN_META_PATH, config.VAL_META_PATH, config.TEST_META_PATH
    )

    # 3. Model Initialization
    print(f"Initializing model: {config.MODEL_NAME}")
    model = model_lib.OrdinalModel().to(device)

    # 4. Training Configuration
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    # BCEWithLogitsLoss is used for the ordinal outputs (binary tasks)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Train Model
    # Note: We use the config values. For a fast baseline, the config defines 12 epochs
    # which is appropriate for this dataset size (~2600 images).
    engine.train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        config.NUM_EPOCHS,
        config.PATIENCE,
    )

    # 6. Evaluation
    print("\nLoading best model for evaluation...")
    utils.load_checkpoint(model)

    preds, targets, ids = get_validation_predictions(model, val_loader, device)

    # Compute Metric
    final_metric = utils.quadratic_weighted_kappa(targets, preds)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    perform_failure_analysis(config.VAL_META_PATH, ids, preds, targets)

    # 8. Submission
    threshold = 0.9194950903896975
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )
        engine.predict_and_submit(model, test_loader, device, config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
