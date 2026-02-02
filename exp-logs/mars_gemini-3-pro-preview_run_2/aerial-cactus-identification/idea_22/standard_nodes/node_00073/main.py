import os
import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr
from library import config, utils, engine, dataset, model


def main():
    # 1. Setup
    utils.set_seed(42)
    device = config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    # Using cached data as requested to speed up loading
    dataloaders = dataset.get_dataloaders(load_cached_data=True)

    # 3. Training Loop (5 Seeds)
    seed_aucs = []
    for seed in config.SEEDS:
        # Train the model for this seed
        # engine.train_seed handles model init, training, and saving checkpoint
        print(f"\n--- Training Seed {seed} ---")
        best_auc = engine.train_seed(seed, dataloaders, device, epochs=config.EPOCHS)
        seed_aucs.append(best_auc)

    print(f"\nAverage Single-Seed Validation AUC: {np.mean(seed_aucs):.6f}")

    # 4. Ensemble Validation
    print("\nRunning Ensemble Validation...")
    val_loader = dataloaders["val"]

    # Accumulate probabilities from all models
    ensemble_probs = None
    targets = None

    # We iterate through each trained seed model
    for seed in config.SEEDS:
        model_filename = f"model_seed_{seed}.pth"

        # Initialize model architecture
        net = model.WideSERes2Net().to(device)

        # Load weights
        try:
            utils.load_checkpoint(net, model_filename, device=device)
        except FileNotFoundError:
            print(f"Warning: Model for seed {seed} not found. Skipping in ensemble.")
            continue

        net.eval()

        seed_preds = []
        seed_targets = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)

                # Forward pass
                outputs = net(images)
                probs = torch.sigmoid(outputs)

                seed_preds.append(probs.cpu().numpy())

                # Collect targets only once (from the first seed loop)
                if targets is None:
                    seed_targets.append(labels.numpy())

        # Concatenate batches
        seed_preds = np.concatenate(seed_preds)

        # Initialize or Add to ensemble
        if ensemble_probs is None:
            ensemble_probs = seed_preds
            # Flatten targets
            targets = np.concatenate(seed_targets)
        else:
            ensemble_probs += seed_preds

    # Average the probabilities
    if ensemble_probs is not None:
        avg_probs = ensemble_probs / len(config.SEEDS)

        # Flatten arrays for metric calculation (Handle (N, 1) vs (N,))
        targets = targets.flatten()
        avg_probs = avg_probs.flatten()

        # Calculate Final Metric
        final_auc = utils.calculate_roc_auc(targets, avg_probs)
        print(f"Final Validation Metric: {final_auc}")
    else:
        print("Error: No predictions generated during validation.")
        return

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")

    # Calculate Error Magnitude: |Target - Prediction|
    # Target is 0 or 1, Prediction is [0, 1]
    error_magnitude = np.abs(targets - avg_probs)

    # Extract Image Meta-Features from Validation Set
    # We iterate the loader again to get images.
    # Loader is deterministic (shuffle=False), so order matches 'error_magnitude'.

    meta_brightness = []
    meta_contrast = []
    meta_red = []
    meta_green = []
    meta_blue = []

    with torch.no_grad():
        for images, _, _ in val_loader:
            # images is Tensor (B, C, H, W) in [0, 1]
            imgs_np = images.numpy()

            for img in imgs_np:
                # img shape (3, 32, 32)
                # Global stats
                meta_brightness.append(np.mean(img))
                meta_contrast.append(np.std(img))

                # Channel stats
                meta_red.append(np.mean(img[0]))
                meta_green.append(np.mean(img[1]))
                meta_blue.append(np.mean(img[2]))

    # Convert to arrays
    features = {
        "Brightness": np.array(meta_brightness),
        "Contrast": np.array(meta_contrast),
        "Red Mean": np.array(meta_red),
        "Green Mean": np.array(meta_green),
        "Blue Mean": np.array(meta_blue),
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, feat_values in features.items():
        if len(feat_values) != len(error_magnitude):
            print(f"Shape mismatch: {len(feat_values)} vs {len(error_magnitude)}")
            continue

        corr, _ = pearsonr(error_magnitude, feat_values)
        print(f"{name}: {corr:.4f}")

    # 6. Submission
    # We generate the submission file using the ensemble of trained models.
    engine.run_inference_ensemble(dataloaders, device)


if __name__ == "__main__":
    main()
