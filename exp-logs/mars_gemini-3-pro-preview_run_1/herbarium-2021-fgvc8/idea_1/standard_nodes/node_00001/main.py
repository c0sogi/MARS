import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2
from sklearn.metrics import f1_score
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.trainer import Trainer
from library.inference import predict_and_submit
from library.dataset import get_label_mapping


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup and Configuration Override
    set_seed(Config.SEED)

    # Override Config for Fast Baseline
    # We use 3 epochs and a large batch size for speed on A100
    Config.NUM_EPOCHS = 3
    Config.BATCH_SIZE = 512

    # 2. Prepare Data
    print("Preparing data...")
    # Load full training metadata
    full_train_df = pd.read_csv("./metadata/train.csv")

    # Generate/Cache label mapping using the FULL dataset first.
    # This ensures that even if we subsample, the mapping covers all classes (including singletons)
    # preventing KeyErrors during validation/inference if a class is missing from the subsample.
    # We force regeneration to ensure it matches the full dataset.
    get_label_mapping(full_train_df, load_cached_data=False)

    # Subsample training data to ~300,000 samples to ensure training completes quickly
    # This is approximately 20% of the dataset.
    SAMPLE_SIZE = 300000
    if len(full_train_df) > SAMPLE_SIZE:
        print(f"Subsampling training data to {SAMPLE_SIZE} samples...")
        train_sub = full_train_df.sample(
            n=SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)

        # Save subsampled csv to working directory
        sub_csv_path = os.path.join(Config.WORKING_DIR, "train_sub.csv")
        train_sub.to_csv(sub_csv_path, index=False)

        # Override Config to use the subsampled CSV for training
        Config.TRAIN_CSV = sub_csv_path
    else:
        print("Using full training data (smaller than sample limit).")

    # 3. Training
    print("Initializing Trainer...")
    trainer = Trainer(debug=False)

    print("Starting Training...")
    trainer.train()

    # 4. Final Validation Evaluation
    print("Evaluating on full validation set...")
    # We perform inference on the validation set to calculate the metric and for failure analysis
    trainer.model.eval()
    val_loader = trainer.val_loader
    device = trainer.device

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # Use autocast for speed
            with torch.cuda.amp.autocast():
                outputs = trainer.model(images)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate Metric
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Final Validation Metric: {macro_f1}")

    # 5. Failure Analysis
    print("Performing failure analysis...")
    # We analyze a subset of the validation set to save time on I/O
    ANALYSIS_COUNT = 2000

    # Load validation metadata to get file paths
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment: DataLoader is sequential (shuffle=False), so val_df rows match all_preds order
    # Slice the first N samples
    analysis_df = val_df.iloc[:ANALYSIS_COUNT].copy()
    analysis_preds = all_preds[:ANALYSIS_COUNT]
    analysis_labels = all_labels[:ANALYSIS_COUNT]

    # Calculate Error (1 if incorrect, 0 if correct)
    # Note: analysis_labels are mapped indices, analysis_preds are mapped indices.
    analysis_df["predicted_idx"] = analysis_preds
    analysis_df["true_idx"] = analysis_labels
    analysis_df["error"] = (
        analysis_df["predicted_idx"] != analysis_df["true_idx"]
    ).astype(int)

    # Extract Image Features (Width, Height, Aspect Ratio)
    widths = []
    heights = []

    for _, row in analysis_df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # We use cv2 to read the image
            img = cv2.imread(img_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(np.nan)
                heights.append(np.nan)
        except Exception:
            widths.append(np.nan)
            heights.append(np.nan)

    analysis_df["width"] = widths
    analysis_df["height"] = heights
    analysis_df["aspect_ratio"] = analysis_df["width"] / analysis_df["height"]

    # Drop rows where image loading failed
    analysis_df = analysis_df.dropna(subset=["width", "height"])

    # Calculate Correlations
    # We look for correlation between Error and features
    if len(analysis_df) > 0:
        correlations = analysis_df[["error", "width", "height", "aspect_ratio"]].corr()[
            "error"
        ]
        print("Correlation between Error and Input Features:")
        print(correlations)
    else:
        print("Not enough data for failure analysis.")

    # 6. Submission
    print("Generating submission file...")
    # predict_and_submit uses the best saved model from training
    predict_and_submit(debug=False, model_path=trainer.model_path)


if __name__ == "__main__":
    main()
