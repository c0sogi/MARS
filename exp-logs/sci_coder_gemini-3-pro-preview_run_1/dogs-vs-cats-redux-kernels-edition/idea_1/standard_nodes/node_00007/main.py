import os
import cv2
import numpy as np
import pandas as pd
import torch
import warnings
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import provided library components
from library.utils import Config, set_seed
from library.engine import train_model, predict_and_submit
from library.dataset import create_dataloaders

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Configuration
    # Using updated config defaults (epochs=3, img_size=256)
    config = Config(seed=42, debug=False)
    set_seed(config.seed)

    print("--- Starting Pipeline ---")

    # 2. Train Model
    # train_model handles the training loop, early stopping, and saving the best model.
    # It returns the model with the best weights loaded.
    model = train_model(config)

    # 3. Validation Assessment
    print("\n--- Validation Assessment ---")

    # Create dataloaders to get the validation set
    loaders = create_dataloaders(config)
    val_loader = loaders["val"]

    device = config.device
    model.eval()

    all_preds = []
    all_labels = []

    # Inference on validation set
    # We disable gradients for speed and memory efficiency
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Use mixed precision for inference speed
            with torch.cuda.amp.autocast():
                logits = model(images)
                probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate Log Loss
    # We clip predictions slightly to avoid log(0) issues, though sklearn handles this usually.
    metric = log_loss(all_labels, all_preds, labels=[0, 1])

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Load validation metadata to access file paths
    df_val = pd.read_csv(config.val_metadata)

    # Ensure alignment: The DataLoader does not shuffle val set, so order matches df_val
    if len(df_val) != len(all_preds):
        print("Warning: Mismatch between validation dataframe and predictions length.")

    # Calculate Error Magnitude (L1 Error)
    errors = np.abs(all_labels - all_preds)

    # Extract input features for correlation analysis
    widths = []
    heights = []
    file_sizes = []

    input_dir = config.input_dir

    # Iterate through validation files to extract metadata
    # This is relatively fast for 4500 images
    for idx, row in df_val.iterrows():
        full_path = os.path.join(input_dir, row["filepath"])

        try:
            # Get file size
            f_size = os.path.getsize(full_path)
            file_sizes.append(f_size)

            # Get dimensions
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(np.nan)
                heights.append(np.nan)
        except Exception:
            file_sizes.append(np.nan)
            widths.append(np.nan)
            heights.append(np.nan)

    # Create analysis DataFrame
    analysis_df = pd.DataFrame(
        {
            "error": errors[
                : len(widths)
            ],  # Ensure length match if something went wrong
            "width": widths,
            "height": heights,
            "file_size": file_sizes,
        }
    )

    # Drop any failed reads
    analysis_df = analysis_df.dropna()

    print("Correlation between Error Magnitude and Input Features:")
    features = ["width", "height", "file_size"]

    for feat in features:
        if len(analysis_df) > 1:
            # Pearson correlation
            corr, _ = pearsonr(analysis_df["error"], analysis_df[feat])
            print(f"  {feat}: {corr:.6f}")
        else:
            print(f"  {feat}: Insufficient data")

    # 5. Submission
    baseline_metric = 0.028030244588095995
    if metric < baseline_metric:
        print("\n--- Generating Submission ---")
        predict_and_submit(config)
    else:
        print(
            f"\nMetric {metric:.5f} did not improve upon baseline {baseline_metric:.5f}. Skipping submission."
        )

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
