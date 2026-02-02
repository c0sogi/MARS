import sys
import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from sklearn.metrics import roc_auc_score

# Add current directory to path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, calculate_class_weights
from library.dataset import get_loaders
from library.models import get_model
from library.engine import train_one_epoch, validate_one_epoch
from library.production import train_final_model
from library.inference import predict_and_submit


def run():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print("Starting Orchestration...")

    # 2. Calibration Phase
    # We manually implement the loop to capture metrics and predictions for failure analysis
    num_epochs = Config.EPOCHS_CALIBRATION
    n_folds = Config.N_FOLDS

    # Load full data for weights calculation
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_full = pd.concat([df_train, df_val], ignore_index=True)

    class_weights_np = calculate_class_weights(
        df_full, Config.TARGET_COLS, load_cached_data=True
    )
    class_weights = torch.tensor(class_weights_np).to(device)
    print(f"Class Weights: {class_weights_np}")

    # Track metrics: [Fold, Epoch]
    fold_auc_history = np.zeros((n_folds, num_epochs))

    # Storage for Failure Analysis (Fold 0)
    # fold0_data[epoch] = {'preds': ..., 'targets': ..., 'df': ...}
    fold0_data = {}

    for fold in range(n_folds):
        print(f"\n[Calibration] Fold {fold + 1}/{n_folds}")

        train_loader, val_loader = get_loaders(fold=fold, mode="calibration")

        model = get_model(pretrained=True)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )
        # Using WarmRestarts as per Lesson 00003
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=num_epochs, T_mult=1, eta_min=Config.MIN_LR
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, criterion, scheduler
            )

            # Validation
            if fold == 0:
                # Custom validation for Fold 0 to capture predictions for failure analysis
                model.eval()
                preds_list = []
                targets_list = []
                running_loss = 0.0
                total_samples = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                        outputs = model(inputs)
                        loss = criterion(outputs, torch.argmax(targets, dim=1))

                        running_loss += loss.item() * inputs.size(0)
                        total_samples += inputs.size(0)

                        probs = torch.softmax(outputs, dim=1)
                        preds_list.append(probs.cpu().numpy())
                        targets_list.append(targets.cpu().numpy())

                epoch_loss = running_loss / total_samples
                preds_array = np.vstack(preds_list)
                targets_array = np.vstack(targets_list)

                try:
                    val_auc = roc_auc_score(
                        targets_array, preds_array, average="macro", multi_class="ovr"
                    )
                except Exception as e:
                    print(f"AUC Calc failed: {e}")
                    val_auc = 0.0

                print(
                    f"Fold {fold+1} Epoch {epoch+1} - Train Loss: {train_loss:.4f} Val Loss: {epoch_loss:.4f} Val AUC: {val_auc:.4f}"
                )

                # Store data for failure analysis
                fold0_data[epoch] = {
                    "preds": preds_array,
                    "targets": targets_array,
                    "df": val_loader.dataset.df.copy(),
                }

            else:
                # Standard validation for other folds
                val_loss, val_auc = validate_one_epoch(
                    model, val_loader, device, criterion
                )
                print(
                    f"Fold {fold+1} Epoch {epoch+1} - Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val AUC: {val_auc:.4f}"
                )

            fold_auc_history[fold, epoch] = val_auc

        # Cleanup to save memory
        del model, optimizer, scheduler, criterion, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # 3. Analysis & Optimal Epoch
    # Calculate mean AUC across folds for each epoch
    mean_auc_per_epoch = np.mean(fold_auc_history, axis=0)
    best_epoch_idx = np.argmax(mean_auc_per_epoch)
    optimal_epochs = int(best_epoch_idx + 1)
    best_auc = mean_auc_per_epoch[best_epoch_idx]

    print(f"\nOptimal Epochs: {optimal_epochs} (Mean AUC: {best_auc})")
    print(f"Final Validation Metric: {best_auc}")

    # 4. Failure Analysis
    print("\n==== Failure Analysis (Fold 0 Validation Set) ====")
    # Retrieve data from the optimal epoch of Fold 0
    f0_info = fold0_data[best_epoch_idx]
    preds = f0_info["preds"]
    targets = f0_info["targets"]
    df_val_fold0 = f0_info["df"]

    # Error = 1 - probability assigned to the true class
    # Targets are one-hot, so sum(targets * preds) gives the prob of the correct class
    true_class_probs = np.sum(targets * preds, axis=1)
    errors = 1.0 - true_class_probs
    df_val_fold0["error"] = errors

    # Compute metadata stats for correlation
    widths = []
    heights = []
    intensities = []

    print("Computing image statistics for failure analysis...")
    for idx, row in df_val_fold0.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        # Read image to get stats
        img = cv2.imread(full_path)
        if img is not None:
            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            # Intensity
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            intensities.append(img_rgb.mean())
        else:
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    df_val_fold0["width"] = widths
    df_val_fold0["height"] = heights
    df_val_fold0["intensity"] = intensities

    # Calculate Correlations
    corr_width = df_val_fold0["error"].corr(df_val_fold0["width"])
    corr_height = df_val_fold0["error"].corr(df_val_fold0["height"])
    corr_intensity = df_val_fold0["error"].corr(df_val_fold0["intensity"])

    print(f"Correlation between Error and Width: {corr_width:.10f}")
    print(f"Correlation between Error and Height: {corr_height:.10f}")
    print(f"Correlation between Error and Intensity: {corr_intensity:.10f}")

    # 5. Submission Logic
    threshold = 0.9871488489626378
    if best_auc > threshold:
        print(
            f"\nMetric {best_auc} > {threshold}. Proceeding to Production Training and Submission."
        )
        # Train final model on 100% data
        final_model = train_final_model(optimal_epochs)
        # Generate submission
        predict_and_submit(final_model)
    else:
        print(f"\nMetric {best_auc} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    run()
