import sys
import os
import torch
import torch.optim as optim
import numpy as np
import cv2
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_loaders
from library.model import get_model
from library.engine import fit, validate, predict


def main():
    # --- 1. Setup & Configuration ---
    seed_everything(Config.SEED)
    device = get_device()

    # Adjust Config for Fast Baseline Execution
    # Reducing epochs to 10 ensures completion within 2 hours on A100
    # while providing enough training to hit the high AUC target.
    Config.NUM_EPOCHS = 10

    print(f"Execution Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {Config.NUM_EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Input Dir: {Config.INPUT_DIR}")

    # --- 2. Data Loading ---
    print("\nInitializing Data Loaders...")
    # We use the full dataset (debug=False) to ensure we can reach the target AUC.
    train_loader, val_loader, test_loader = get_loaders(
        debug=False, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # --- 3. Model Initialization ---
    print("\nInitializing Model...")
    model = get_model(device)

    # Optimizer: AdamW with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    # --- 4. Training ---
    print("\nStarting Training Loop...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
    )

    # --- 5. Final Validation ---
    print("\nPerforming Final Validation...")

    # Load the best model checkpoint saved during training
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: No best model checkpoint found. Using current model state.")

    # Calculate final metric on the full validation set
    _, val_auc = validate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # --- 6. Conditional Logic ---
    THRESHOLD = 0.9889066475479729

    if val_auc > THRESHOLD:
        print(
            f"\nMetric ({val_auc:.6f}) > Threshold ({THRESHOLD:.6f}). Proceeding with Analysis and Submission."
        )

        # --- 7. Failure Analysis ---
        print("\n--- Failure Analysis ---")

        # Collect predictions and targets for the validation set
        model.eval()
        all_preds = []
        all_targets = []

        # We also need file paths to load images for feature extraction
        # Accessing the dataframe from the dataset
        val_df = val_loader.dataset.df

        print("Collecting validation predictions...")
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images).view(-1)
                probs = torch.sigmoid(outputs)

                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Calculate Error Magnitude
        errors = np.abs(all_targets - all_preds)

        # Sample a subset for feature extraction to keep it fast (e.g., 2000 images)
        sample_size = min(2000, len(errors))
        indices = np.random.choice(len(errors), size=sample_size, replace=False)

        sampled_errors = errors[indices]
        sampled_paths = val_df.iloc[indices]["file_path"].values

        print(f"Analyzing {sample_size} sampled validation images...")

        # Feature accumulators
        features = {
            "brightness": [],
            "contrast": [],
            "red_mean": [],
            "green_mean": [],
            "blue_mean": [],
        }

        for rel_path in sampled_paths:
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            img = cv2.imread(full_path)

            if img is None:
                # Fallback for missing images
                for k in features:
                    features[k].append(0.0)
                continue

            # Convert to RGB and Normalize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Extract Features
            mean_rgb = img.mean(axis=(0, 1))  # [R, G, B]
            std_rgb = img.std(axis=(0, 1))

            features["brightness"].append(mean_rgb.mean())
            features["contrast"].append(img.std())  # Global contrast
            features["red_mean"].append(mean_rgb[0])
            features["green_mean"].append(mean_rgb[1])
            features["blue_mean"].append(mean_rgb[2])

        # Calculate Correlations
        print("Correlation between Error Magnitude and Input Features:")
        for name, values in features.items():
            # Use NumPy for correlation
            if len(values) > 1:
                corr = np.corrcoef(sampled_errors, values)[0, 1]
                print(f"  {name}: {corr:.4f}")
            else:
                print(f"  {name}: N/A")

        # --- 8. Inference & Submission ---
        print("\nGenerating Submission...")
        predict(model, test_loader, device)

    else:
        print(
            f"\nMetric ({val_auc:.6f}) <= Threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
