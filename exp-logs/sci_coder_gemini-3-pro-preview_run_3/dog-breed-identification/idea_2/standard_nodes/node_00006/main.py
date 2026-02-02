import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from scipy.stats import pearsonr
import torch.nn.functional as F

# Import provided library modules
from library.config import Config
from library import train, predict, utils, dataset, model


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Set seeds for reproducibility
    utils.set_seed(Config.SEED)

    # Override Config for Improved Training
    # Extending fine-tuning to 20 epochs allows the model to fully leverage the
    # pre-trained features and the new augmentations. Cite solution_lesson_node_00005
    Config.WARMUP_EPOCHS = 1
    Config.FINE_TUNE_EPOCHS = 20

    print("Configuration configured for fast baseline:")
    print(f"  Warmup Epochs: {Config.WARMUP_EPOCHS}")
    print(f"  Fine-tune Epochs: {Config.FINE_TUNE_EPOCHS}")
    print(f"  Device: {Config.DEVICE}")

    # ==========================================
    # 2. Training
    # ==========================================
    print("\nStarting Training Pipeline...")
    best_model_path = train.run_training()

    # ==========================================
    # 3. Validation Assessment
    # ==========================================
    print("\nStarting Validation Assessment...")

    # Load validation data
    dataloaders, class_names = dataset.get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )
    val_loader = dataloaders["val"]

    # Load the best model
    device = torch.device(Config.DEVICE)
    net = model.get_model(num_classes=Config.NUM_CLASSES, pretrained=False)

    # Load weights
    checkpoint = torch.load(best_model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        net.load_state_dict(checkpoint["model_state_dict"])
    else:
        net.load_state_dict(checkpoint)

    net = net.to(device)
    net.eval()

    # Inference on Validation Set
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # Calculate Metric
    # y_true are indices, y_pred are probabilities
    val_metric = utils.calculate_log_loss(
        y_true, y_pred, labels=list(range(Config.NUM_CLASSES))
    )

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\nStarting Failure Analysis...")

    # 1. Calculate per-sample error (Log Loss)
    # Extract the probability assigned to the true class
    # y_pred is (N, C), y_true is (N,)
    n_samples = y_true.shape[0]
    true_class_probs = y_pred[np.arange(n_samples), y_true]

    # Clip probabilities to avoid log(0)
    eps = 1e-15
    true_class_probs = np.clip(true_class_probs, eps, 1 - eps)

    # Error magnitude = -log(p_true)
    errors = -np.log(true_class_probs)

    # 2. Extract Input Features
    # We need to load the metadata to get file paths, then read image stats
    # dataset.process_data caches metadata, we can read the cache or the csv
    val_meta_path = Config.VAL_METADATA_PATH
    df_val = pd.read_csv(val_meta_path)

    # Ensure alignment: val_loader (shuffle=False) preserves order of df_val
    if len(df_val) != n_samples:
        print("Warning: Validation dataframe length mismatch with predictions.")

    file_sizes = []
    widths = []
    heights = []

    # Iterate through metadata to extract features
    # This might take a moment, but for ~1800 images it's fast enough
    for _, row in df_val.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            # File Size
            f_size = os.path.getsize(full_path)

            # Image Dimensions
            # We use PIL to open lazily to get size
            with Image.open(full_path) as img:
                w, h = img.size

            file_sizes.append(f_size)
            widths.append(w)
            heights.append(h)
        except Exception:
            # Fallback for missing files
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)

    file_sizes = np.array(file_sizes)
    widths = np.array(widths)
    heights = np.array(heights)

    # 3. Calculate Correlations
    # Handle potential constant values (std=0) to avoid warnings
    def safe_correlation(x, y, name):
        if np.std(x) == 0 or np.std(y) == 0:
            print(f"Correlation (Error vs {name}): Undefined (zero variance)")
        else:
            corr, _ = pearsonr(x, y)
            print(f"Correlation (Error vs {name}): {corr:.4f}")

    safe_correlation(file_sizes, errors, "File Size")
    safe_correlation(widths, errors, "Width")
    safe_correlation(heights, errors, "Height")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.22554498779858895

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({val_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        predict.generate_submission(model_path=best_model_path)
    else:
        print(
            f"\nValidation metric ({val_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
