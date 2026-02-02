import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Import provided library components
from library.utils import set_seed, get_device, WORKING_DIR, SUBMISSION_DIR
from library.model import WideSEResNet
from library.dataset import get_dataloaders
from library.engine import train_one_epoch, evaluate, predict_with_tta

# Configuration for Fast Baseline
EPOCHS = 20
BATCH_SIZE = 64
SEEDS = [0, 1, 2, 3, 4]


def main():
    # Setup
    device = get_device()
    print(f"Using device: {device}")

    # Load Data
    # load_cached_data=True to use preprocessed .npy files if available
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=True
    )

    # Extract Validation Targets (for Ensemble Metric)
    # val_loader has shuffle=False, so order is deterministic
    val_targets = []
    for _, labels in val_loader:
        val_targets.extend(labels.numpy())
    val_targets = np.array(val_targets)

    # Extract Test IDs
    # test_loader.dataset.targets contains the IDs for the test set
    test_ids = test_loader.dataset.targets

    # Placeholders for Ensemble Predictions
    # We sum probabilities from each seed, then divide by N_SEEDS
    ensemble_val_probs = np.zeros(len(val_targets))
    ensemble_test_probs = np.zeros(len(test_ids))

    # --- Training Loop ---
    for seed in SEEDS:
        print(f"\n--- Processing Seed {seed} ---")
        set_seed(seed)

        # Initialize Model
        model = WideSEResNet().to(device)

        # Setup Training Components
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # Training
        best_auc = 0.0
        best_model_path = os.path.join(WORKING_DIR, f"model_seed_{seed}.pth")

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            # Save Best
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

        print(f"Best AUC for Seed {seed}: {best_auc:.6f}")

        # --- Inference ---
        # Load best model
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        # 1. Validation Inference (Accumulate for Ensemble)
        seed_val_probs = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                seed_val_probs.extend(probs)
        ensemble_val_probs += np.array(seed_val_probs)

        # 2. Test Inference (Accumulate for Ensemble)
        seed_test_probs = []
        # Iterate test_loader (batch_size=1)
        for images, _ in test_loader:
            # images shape: (1, C, H, W) -> squeeze to (C, H, W) for TTA function
            img_tensor = images.squeeze(0)
            prob = predict_with_tta(model, img_tensor, device)
            seed_test_probs.append(prob)
        ensemble_test_probs += np.array(seed_test_probs)

    # --- Ensemble Aggregation ---
    ensemble_val_probs /= len(SEEDS)
    ensemble_test_probs /= len(SEEDS)

    # --- Final Evaluation ---
    final_val_auc = roc_auc_score(val_targets, ensemble_val_probs)
    print(f"Final Validation Metric: {final_val_auc}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude
    errors = np.abs(val_targets - ensemble_val_probs)

    # Extract features from validation set
    # We iterate val_loader again. Order is preserved (shuffle=False).
    meta_brightness = []
    meta_contrast = []
    meta_red = []
    meta_green = []
    meta_blue = []

    for images, _ in val_loader:
        # images: (B, C, H, W)
        # Compute stats on CPU numpy arrays
        imgs_np = images.numpy()

        # Brightness: Mean of all pixels
        meta_brightness.extend(np.mean(imgs_np, axis=(1, 2, 3)))

        # Contrast: Std of all pixels
        meta_contrast.extend(np.std(imgs_np, axis=(1, 2, 3)))

        # Channel Means
        meta_red.extend(np.mean(imgs_np[:, 0, :, :], axis=(1, 2)))
        meta_green.extend(np.mean(imgs_np[:, 1, :, :], axis=(1, 2)))
        meta_blue.extend(np.mean(imgs_np[:, 2, :, :], axis=(1, 2)))

    # Compute Correlations
    features = {
        "brightness": meta_brightness,
        "contrast": meta_contrast,
        "red_mean": meta_red,
        "green_mean": meta_green,
        "blue_mean": meta_blue,
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, feats in features.items():
        # Use numpy for correlation to avoid extra dependencies
        # np.corrcoef returns matrix [[1, r], [r, 1]]
        corr = np.corrcoef(errors, feats)[0, 1]
        print(f"{name}: {corr:.4f}")

    # --- Submission ---
    # Generate submission file
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    sub_df = pd.DataFrame({"id": test_ids, "has_cactus": ensemble_test_probs})
    sub_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
