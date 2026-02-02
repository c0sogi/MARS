import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_multilabel_auc
from library.dataset import prepare_data, load_test_data, BirdDataset, get_transforms
from library.model import BirdClassifier
from library.trainer import run_fold


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation
    # Load training metadata and create/load 5-fold CV splits
    full_df = prepare_data(load_cached_data=True)

    # Containers for Out-Of-Fold (OOF) predictions
    # We will average predictions from all backbones for each sample
    num_samples = len(full_df)
    oof_preds_sum = np.zeros((num_samples, Config.NUM_CLASSES))
    oof_counts = np.zeros((num_samples, 1))

    # Store trained model paths for final inference
    all_model_paths = []

    # 3. Training and Validation Loop
    # Iterate through each backbone architecture and each fold
    for backbone in Config.BACKBONES:
        for fold_idx in range(Config.N_FOLDS):
            # Train the model for this fold
            # run_fold handles training, checkpointing, and averaging top-k checkpoints
            model_path = run_fold(fold_idx, full_df, backbone)
            all_model_paths.append((backbone, model_path))

            # --- Generate OOF Predictions ---
            # Load the trained (averaged) model
            model = BirdClassifier(backbone, Config.NUM_CLASSES, pretrained=False)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            # Prepare validation data for this fold
            val_df = full_df[full_df["fold"] == fold_idx]
            val_indices = val_df.index.values

            val_dataset = BirdDataset(
                val_df, transforms=get_transforms("valid"), mode="train"
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Inference
            fold_preds = []
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = torch.sigmoid(outputs)
                    fold_preds.append(preds.cpu().numpy())

            if len(fold_preds) > 0:
                fold_preds = np.concatenate(fold_preds, axis=0)

                # Accumulate predictions
                oof_preds_sum[val_indices] += fold_preds
                oof_counts[val_indices] += 1

            # Cleanup to save memory
            del model
            torch.cuda.empty_cache()

    # 4. Calculate Final Validation Metric
    # Average the OOF predictions across the ensemble (3 backbones)
    # Avoid division by zero
    oof_counts[oof_counts == 0] = 1
    final_oof_preds = oof_preds_sum / oof_counts

    # Construct ground truth matrix
    y_true = np.zeros((num_samples, Config.NUM_CLASSES))
    for idx, row in full_df.iterrows():
        label_str = row["labels"]
        if (
            isinstance(label_str, str)
            and label_str != "?"
            and len(label_str.strip()) > 0
        ):
            indices = [int(x) for x in label_str.split()]
            y_true[idx, indices] = 1

    # Calculate AUC
    final_auc = calculate_multilabel_auc(y_true, final_oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate Binary Cross Entropy Loss per sample as a proxy for error magnitude
    # Clip predictions to avoid log(0)
    eps = 1e-7
    preds_clipped = np.clip(final_oof_preds, eps, 1 - eps)
    bce_per_sample = -(
        y_true * np.log(preds_clipped) + (1 - y_true) * np.log(1 - preds_clipped)
    )
    # Mean loss across classes for each sample
    error_magnitude = np.mean(bce_per_sample, axis=1)

    # Feature 1: Number of Labels (Complexity)
    label_counts = np.sum(y_true, axis=1)

    # Feature 2: Image Brightness (Signal Energy proxy)
    # Quickly compute mean pixel intensity for each spectrogram
    image_means = []
    for idx, row in full_df.iterrows():
        wav_path = row["file_path"]
        filename = os.path.basename(wav_path).replace(".wav", ".bmp")
        image_path = os.path.join(Config.SPECTROGRAM_DIR, filename)
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                image_means.append(np.mean(img))
            else:
                image_means.append(0)
        except Exception:
            image_means.append(0)
    image_means = np.array(image_means)

    # Calculate Correlations
    if len(error_magnitude) > 1:
        corr_labels, _ = pearsonr(error_magnitude, label_counts)
        corr_brightness, _ = pearsonr(error_magnitude, image_means)
        print(f"Correlation (Error vs Label Count): {corr_labels:.4f}")
        print(f"Correlation (Error vs Image Brightness): {corr_brightness:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # 6. Submission Generation
    THRESHOLD = 0.9479806884980326

    if final_auc > THRESHOLD:
        print(
            f"Validation metric {final_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        test_df = load_test_data()
        test_dataset = BirdDataset(
            test_df, transforms=get_transforms("test"), mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Ensemble Inference on Test Set
        test_preds_sum = np.zeros((len(test_df), Config.NUM_CLASSES))

        for backbone, model_path in all_model_paths:
            model = BirdClassifier(backbone, Config.NUM_CLASSES, pretrained=False)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            model_preds = []
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = torch.sigmoid(outputs)
                    model_preds.append(preds.cpu().numpy())

            if len(model_preds) > 0:
                test_preds_sum += np.concatenate(model_preds, axis=0)

            del model
            torch.cuda.empty_cache()

        # Average predictions
        avg_test_preds = test_preds_sum / len(all_model_paths)

        # Format Submission
        submission_rows = []
        for idx, row in test_df.iterrows():
            rec_id = row["rec_id"]
            probs = avg_test_preds[idx]
            for species_id, prob in enumerate(probs):
                # Id format: rec_id * 100 + species_id
                submission_id = int(rec_id * 100 + species_id)
                submission_rows.append({"Id": submission_id, "Probability": prob})

        sub_df = pd.DataFrame(submission_rows)
        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"Validation metric {final_auc} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
