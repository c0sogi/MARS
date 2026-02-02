import os
import numpy as np
import pandas as pd
import torch
import cv2

from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.dataset import get_dataloaders
from library.model import AppleClassifier
from library.loss import WeightedLabelSmoothCrossEntropy
from library.train import train_one_epoch, valid_one_epoch, inference


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Load cached data to save time if available
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Class Weights Calculation
    # Extract labels from the training dataset to compute imbalance-aware weights
    train_labels = train_loader.dataset.labels
    class_counts = np.sum(train_labels, axis=0)
    num_samples = len(train_labels)
    num_classes = train_labels.shape[1]

    # Inverse Class Frequency Weights
    class_weights = num_samples / (num_classes * (class_counts + 1e-6))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights}")

    # 4. Model Initialization
    model = AppleClassifier(pretrained=True)
    model.to(device)

    # 5. Loss, Optimizer, Scheduler
    criterion = WeightedLabelSmoothCrossEntropy(
        weight=class_weights_tensor, smoothing=Config.LABEL_SMOOTHING
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 6. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        val_loss, val_auc = valid_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    # 7. Final Validation & Failure Analysis
    print("\nLoading best model for analysis...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model file not found. Using current model weights.")

    model.eval()

    # Generate raw predictions on validation set for analysis
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            val_preds_list.append(probs.cpu().numpy())
            val_targets_list.append(labels.numpy())

    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    # Calculate Final Metric
    final_metric = calculate_metric(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate Mean Absolute Error per sample
    errors = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Extract meta-features by reading original images
    val_df = val_loader.dataset.df
    widths = []
    heights = []
    intensities = []

    print("Performing failure analysis...")
    for idx, row in val_df.iterrows():
        fpath = os.path.join(Config.INPUT_DIR, row["file_path"])
        if os.path.exists(fpath):
            img = cv2.imread(fpath)
            if img is not None:
                h, w, c = img.shape
                widths.append(w)
                heights.append(h)
                intensities.append(img.mean())
            else:
                widths.append(0)
                heights.append(0)
                intensities.append(0)
        else:
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    # Calculate correlations
    def print_corr(name, feat_arr, err_arr):
        if len(feat_arr) > 1 and np.std(feat_arr) > 0:
            corr = np.corrcoef(feat_arr, err_arr)[0, 1]
            print(f"Correlation between Error and {name}: {corr:.4f}")
        else:
            print(f"Correlation between Error and {name}: N/A (Constant feature)")

    print_corr("Width", widths, errors)
    print_corr("Height", heights, errors)
    print_corr("Intensity", intensities, errors)

    # 8. Conditional Submission
    THRESHOLD = 0.9902480620249655
    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference on Test Set (using TTA as per Config)
        image_ids, test_preds = inference(model, test_loader, device)

        # Create submission DataFrame
        # Columns must match dataset.LABEL_COLS order: ["healthy", "multiple_diseases", "rust", "scab"]
        submission_df = pd.DataFrame(
            {
                "image_id": image_ids,
                "healthy": test_preds[:, 0],
                "multiple_diseases": test_preds[:, 1],
                "rust": test_preds[:, 2],
                "scab": test_preds[:, 3],
            }
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
