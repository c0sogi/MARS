import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from PIL import Image
import warnings

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.train import run_training, run_inference
from library.model import AppleDiseaseSwinModel
from library.dataset import AppleDataset, get_transforms
from library.utils import seed_everything


def main():
    # 1. Configuration Setup
    # Override Config defaults to match the "Idea" strategy (256px)
    # and ensure execution within time limits (12 epochs).
    Config.img_size = 256
    Config.epochs = 12
    Config.batch_size = 32  # A100 can handle Swin-Tiny @ 384 with batch 32

    # Ensure reproducibility
    seed_everything(Config.seed)

    print(
        f"Configuration: Model={Config.model_name}, Size={Config.img_size}, Epochs={Config.epochs}"
    )

    # 2. Training
    # Run the training pipeline with explicit hyperparameters
    run_training(
        debug=False,
        epochs=Config.epochs,
        batch_size=Config.batch_size,
        learning_rate=Config.learning_rate,
        weight_decay=Config.weight_decay,
        min_lr=Config.min_lr,
        label_smoothing=Config.label_smoothing,
        num_workers=Config.num_workers,
    )

    # 3. Validation & Failure Analysis
    print("\nStarting Validation and Failure Analysis...")

    # Load Validation Metadata
    df_val = pd.read_csv(Config.val_metadata_path)

    # Setup Validation Dataset & Loader
    val_dataset = AppleDataset(
        df_val, mode="val", transform=get_transforms("val", Config.img_size)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Load the Best Model
    device = Config.device
    model = AppleDiseaseSwinModel(pretrained=False)  # Architecture only

    if not os.path.exists(Config.model_save_path):
        print("Error: Model checkpoint not found.")
        return

    checkpoint = torch.load(Config.model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Inference Loop
    all_probs = []
    all_targets = []

    # Features for failure analysis
    feature_file_size = []
    feature_width = []
    feature_height = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            # Move to device
            images = images.to(device)

            # Forward pass (Mixed Precision for speed)
            with torch.cuda.amp.autocast():
                logits = model(images)

            # Probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())

            # Extract features for this batch (Lazy load metadata)
            start_idx = i * Config.batch_size
            end_idx = min((i + 1) * Config.batch_size, len(df_val))

            for idx in range(start_idx, end_idx):
                rel_path = df_val.iloc[idx]["file_path"]
                full_path = os.path.join(Config.input_dir, rel_path)

                # File Size
                try:
                    fsize = os.path.getsize(full_path)
                except:
                    fsize = 0
                feature_file_size.append(fsize)

                # Dimensions (Use PIL to read header only, fast)
                try:
                    with Image.open(full_path) as img:
                        w, h = img.size
                except:
                    w, h = 0, 0
                feature_width.append(w)
                feature_height.append(h)

    # Concatenate results
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Metric
    # Binarize predictions
    y_pred_binary = (all_probs > Config.threshold).astype(int)

    # Macro F1 Score
    final_metric = f1_score(
        all_targets, y_pred_binary, average="macro", zero_division=0
    )

    # Print Required Metric
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Metric: Mean Absolute Error per sample (average over classes)
    mae_per_sample = np.mean(np.abs(all_probs - all_targets), axis=1)

    print("\n--- Failure Analysis ---")
    print("Correlation between Error Magnitude (MAE) and Input Features:")

    features = {
        "File Size": feature_file_size,
        "Image Width": feature_width,
        "Image Height": feature_height,
    }

    for name, data in features.items():
        if len(data) != len(mae_per_sample):
            print(f"{name}: Size mismatch")
            continue

        # Check for constant values to avoid warning
        if np.std(data) == 0:
            print(f"{name}: Constant value (Corr: N/A)")
        else:
            corr, _ = pearsonr(data, mae_per_sample)
            print(f"{name}: {corr:.6f}")

    # 4. Submission
    TARGET_THRESHOLD = 0.9187550291577454

    if final_metric > TARGET_THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({TARGET_THRESHOLD}). Generating submission..."
        )
        # Explicitly pass batch_size to ensure consistency
        run_inference(batch_size=Config.batch_size, num_workers=Config.num_workers)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({TARGET_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
