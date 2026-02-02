import os
import cv2
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library import utils, data, model, trainer, inference


def main():
    # 1. Setup and Configuration
    # Set seed for reproducibility
    utils.set_seed(Config.SEED)

    # Adjust configuration for a fast baseline execution
    # We limit epochs to ensure quick turnaround while allowing sufficient learning
    Config.EPOCHS = 15

    print("==== Starting Runfile ====")
    print(f"Device: {utils.get_device()}")

    # 2. Training
    print("\n[Step 1] Training Model...")
    # Initialize trainer
    model_trainer = trainer.Trainer(debug=False)
    # Run training loop
    model_trainer.fit()

    # 3. Validation and Metrics
    print("\n[Step 2] Validation Assessment...")

    # Load the best model weights
    device = utils.get_device()
    best_model = model.AppleDiseaseModel(
        model_name=Config.MODEL_NAME,
        pretrained=False,  # Structure only, weights loaded below
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model for validation.")
    else:
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    best_model.eval()

    # Get validation data loader
    # We discard train and test loaders here
    _, val_loader, _ = data.get_dataloaders(debug=False)

    # Inference on validation set with Test-Time Augmentation (TTA)
    all_probs = []
    all_labels = []
    all_errors = []  # To store (1 - prob_of_true_class)

    print("Performing Validation with TTA (Original + HFlip + VFlip)...")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # TTA: 1. Original
            logits_orig = best_model(images)
            probs_orig = torch.softmax(logits_orig, dim=1)

            # TTA: 2. Horizontal Flip
            images_hflip = torch.flip(images, [3])
            logits_hflip = best_model(images_hflip)
            probs_hflip = torch.softmax(logits_hflip, dim=1)

            # TTA: 3. Vertical Flip
            images_vflip = torch.flip(images, [2])
            logits_vflip = best_model(images_vflip)
            probs_vflip = torch.softmax(logits_vflip, dim=1)

            # Average probabilities
            probs = (probs_orig + probs_hflip + probs_vflip) / 3.0

            # Store for metric calculation
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Calculate error magnitude for failure analysis
            batch_probs = probs.cpu().numpy()
            batch_labels = labels.cpu().numpy()

            for i in range(len(batch_labels)):
                true_class_idx = batch_labels[i]
                pred_prob_true = batch_probs[i, true_class_idx]
                # Error magnitude: 0 if perfect prediction (1.0), 1 if completely wrong (0.0)
                error_mag = 1.0 - pred_prob_true
                all_errors.append(error_mag)

    # Concatenate results
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Calculate Final Validation Metric (Mean column-wise ROC AUC)
    try:
        val_auc = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro"
        )
    except Exception as e:
        print(f"Error calculating ROC AUC: {e}")
        val_auc = 0.0

    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    print("\n[Step 3] Failure Analysis...")

    # Load validation metadata to access original images for feature extraction
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure the length matches
    if len(val_df) != len(all_errors):
        print(
            f"Warning: Mismatch in validation set size. DF: {len(val_df)}, Errors: {len(all_errors)}"
        )

    brightness_values = []
    contrast_values = []

    # Iterate through validation images to extract features
    # Note: val_loader is created with shuffle=False, so order is preserved relative to val_df
    for idx, row in val_df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            brightness_values.append(0)
            contrast_values.append(0)
            continue

        # Convert to grayscale for simple stats
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Brightness: Mean pixel intensity
        brightness_values.append(np.mean(gray))
        # Contrast: Standard deviation of pixel intensity
        contrast_values.append(np.std(gray))

    # Calculate correlations
    if len(all_errors) == len(brightness_values):
        corr_brightness, _ = pearsonr(all_errors, brightness_values)
        corr_contrast, _ = pearsonr(all_errors, contrast_values)

        print("Correlation between Error Magnitude and Input Features:")
        print(f"  - Brightness: {corr_brightness:.6f}")
        print(f"  - Contrast:   {corr_contrast:.6f}")
    else:
        print("Skipping correlation analysis due to length mismatch.")

    # 5. Submission
    print("\n[Step 4] Generating Submission...")

    BASELINE_SCORE = 0.9772155669486011
    if val_auc > BASELINE_SCORE:
        print(
            f"Validation AUC ({val_auc:.6f}) > Baseline ({BASELINE_SCORE:.6f}). Generating submission."
        )
        inference.predict_and_submit(debug=False)
    else:
        print(
            f"Validation AUC ({val_auc:.6f}) <= Baseline ({BASELINE_SCORE:.6f}). Skipping submission."
        )

    print("\nRunfile execution complete.")


if __name__ == "__main__":
    main()
