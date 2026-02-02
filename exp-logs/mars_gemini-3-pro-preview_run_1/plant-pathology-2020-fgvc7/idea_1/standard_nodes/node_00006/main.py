import os
import pandas as pd
import numpy as np
import torch
import cv2
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import provided library functions
from library.utils import seed_everything, get_device
from library.dataset import AppleDataset, get_transforms, TARGET_COLS
from library.model import EfficientNetB3
from library.train import train_model
from library.inference import predict_and_submit


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    seed_everything(42)
    device = get_device()

    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/baseline"
    SUBMISSION_PATH = "./submission/submission.csv"

    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # ==========================================
    # 2. Model Training
    # ==========================================
    print("Starting model training...")
    # We use 10 epochs for a fast baseline execution as requested.
    # The dataset is small (~1300 images), so this should take < 5 minutes on GPU.
    best_model_path = train_model(
        train_metadata_path=TRAIN_META,
        val_metadata_path=VAL_META,
        input_dir=INPUT_DIR,
        output_dir=WORKING_DIR,
        epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        seed=42,
    )

    # ==========================================
    # 3. Validation & Metric Calculation
    # ==========================================
    print("Evaluating model on validation set...")

    # Load the best model
    model = ResNet34Baseline(num_classes=len(TARGET_COLS), pretrained=False)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare validation loader
    val_dataset = AppleDataset(
        metadata_path=VAL_META,
        transform=get_transforms("val", image_size=256),
        input_dir=INPUT_DIR,
        mode="val",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    all_probs = []
    all_labels = []

    # Inference loop
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Calculate Mean Column-wise ROC AUC
    # We need one-hot encoded labels for multiclass ROC AUC
    y_true_one_hot = np.eye(len(TARGET_COLS))[all_labels]

    try:
        val_auc = roc_auc_score(
            y_true_one_hot, all_probs, average="macro", multi_class="ovr"
        )
    except Exception as e:
        print(f"Warning: Could not calculate ROC AUC. Error: {e}")
        val_auc = 0.0

    # Print the required metric
    print(f"Final Validation Metric: {val_auc}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("Performing failure analysis...")

    # Calculate error magnitude: 1.0 - Probability assigned to the true class
    # all_labels contains indices of true classes
    # all_probs contains probabilities for all classes
    # We extract the probability of the true class for each sample
    true_class_probs = all_probs[np.arange(len(all_labels)), all_labels]
    error_magnitudes = 1.0 - true_class_probs

    # Extract meta-features from images
    # We iterate through the validation dataframe to get file paths
    val_df = val_dataset.df
    meta_stats = []

    for idx, row in val_df.iterrows():
        img_path = os.path.join(INPUT_DIR, row["file_path"])

        # Default values
        w, h, intensity = 0, 0, 0.0

        if os.path.exists(img_path):
            try:
                # Read image to get stats
                img = cv2.imread(img_path)
                if img is not None:
                    h, w, _ = img.shape
                    # Calculate mean intensity (normalized 0-1)
                    intensity = img.mean() / 255.0
            except Exception:
                pass

        meta_stats.append({"width": w, "height": h, "mean_intensity": intensity})

    meta_df = pd.DataFrame(meta_stats)
    meta_df["error"] = error_magnitudes

    # Calculate correlations
    # Drop columns that might be constant (std=0) to avoid NaNs in correlation matrix
    meta_df = meta_df.loc[:, meta_df.std() > 0]

    if "error" in meta_df.columns:
        correlations = meta_df.corr()["error"].drop("error", errors="ignore")
        print("Correlations between Error Magnitude and Input Features:")
        print(correlations)
    else:
        print("Could not calculate correlations (constant error or features).")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    print("Generating submission file...")
    predict_and_submit(
        model_path=best_model_path,
        test_metadata_path=TEST_META,
        input_dir=INPUT_DIR,
        output_path=SUBMISSION_PATH,
        batch_size=32,
        device=device,
    )
    print("Process complete.")


if __name__ == "__main__":
    main()
