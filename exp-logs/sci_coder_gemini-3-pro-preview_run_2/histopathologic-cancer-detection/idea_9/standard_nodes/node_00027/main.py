import os
import shutil
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.dataset import get_dataloaders
from library.models import PathologyClassifier
from library.training import Trainer, set_seed
from library.inference import InferenceEngine, generate_submission


def main():
    # --- 1. Configuration & Setup ---
    # Override Config for a fast baseline execution
    Config.EPOCHS = 2
    # We will manually control the folds loop to run only 1 fold per model for speed
    TRAIN_DEBUG_SIZE = 5000  # Limit training data for speed

    # Clean checkpoint directory to ensure we only use models trained in this run
    if os.path.exists(Config.CHECKPOINT_DIR):
        shutil.rmtree(Config.CHECKPOINT_DIR)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Starting Fast Baseline Run...")
    print(
        f"Configuration: {Config.EPOCHS} Epochs, {TRAIN_DEBUG_SIZE} Training Samples."
    )

    # --- 2. Data Loading (Training) ---
    print("\nLoading Training Data (Subset)...")
    # Load subset for training
    train_loader_subset, _, _ = get_dataloaders(
        load_cached_data=True, debug_size=TRAIN_DEBUG_SIZE
    )

    # Create a small validation loader for the training loop monitoring (to avoid full val overhead during training)
    _, val_loader_subset, _ = get_dataloaders(load_cached_data=True, debug_size=1000)

    # --- 3. Training Loop ---
    # Train one instance of each backbone (Heterogeneous Ensemble)
    for backbone in Config.MODEL_BACKBONES:
        print(f"\n--- Training Backbone: {backbone} ---")

        # Initialize Model
        model = PathologyClassifier(
            model_name=backbone, num_classes=Config.NUM_CLASSES, pretrained=True
        )
        model.to(device)

        # Initialize Trainer (Fold 0)
        trainer = Trainer(
            model, train_loader_subset, val_loader_subset, device, fold_idx=0
        )

        # Execute Training
        trainer.fit()

        # Cleanup to save memory
        del model, trainer
        torch.cuda.empty_cache()

    # --- 4. Full Validation & Metric ---
    print("\n--- Starting Full Validation ---")
    # Load the FULL validation set for the official metric
    print("Loading Full Validation Data...")
    _, val_loader_full, _ = get_dataloaders(load_cached_data=True, debug_size=None)

    # Initialize Inference Engine (loads best models from checkpoints)
    engine = InferenceEngine(Config.CHECKPOINT_DIR, device)

    # Run Inference
    print("Running inference on validation set...")
    val_ids, val_probs = engine.run_inference(val_loader_full)

    # Collect Ground Truth Labels
    # val_loader_full is deterministic (shuffle=False), so we can iterate to get labels
    val_labels = []
    for _, labels, _ in val_loader_full:
        val_labels.append(labels.numpy())
    val_labels = np.concatenate(val_labels).flatten()

    # Calculate AUC
    final_auc = roc_auc_score(val_labels, val_probs)
    # Print exactly as required
    print(f"Final Validation Metric: {final_auc:.15f}")

    # --- 5. Failure Analysis ---
    print("\n--- Performing Failure Analysis ---")

    # Calculate Error Magnitude
    errors = np.abs(val_labels - val_probs)

    # Compute Image Features on Validation Set
    feature_stats = {
        "brightness": [],
        "contrast": [],
        "red_mean": [],
        "green_mean": [],
        "blue_mean": [],
    }

    print("Computing image features...")
    # Iterate validation loader to compute stats
    for images, _, _ in val_loader_full:
        # images is (B, 3, 64, 64) tensor
        imgs_np = images.numpy()

        # Mean across H, W -> (B, 3)
        means = imgs_np.mean(axis=(2, 3))

        # Brightness: Mean of channels
        b_batch = means.mean(axis=1)
        # Contrast: Std of the whole image (approx)
        c_batch = imgs_np.std(axis=(1, 2, 3))

        feature_stats["brightness"].extend(b_batch)
        feature_stats["contrast"].extend(c_batch)
        feature_stats["red_mean"].extend(means[:, 0])
        feature_stats["green_mean"].extend(means[:, 1])
        feature_stats["blue_mean"].extend(means[:, 2])

    # Calculate and Print Correlations
    print("Correlation between Error Magnitude and Input Features:")
    for feat_name, feat_values in feature_stats.items():
        feat_values = np.array(feat_values)
        if len(feat_values) == len(errors):
            corr, _ = pearsonr(errors, feat_values)
            print(f"  {feat_name}: {corr:.10f}")
        else:
            print(f"  {feat_name}: Error (Shape Mismatch)")

    # --- 6. Submission ---
    THRESHOLD = 0.9889066475479729

    if final_auc > THRESHOLD:
        print(f"\nValidation AUC exceeds {THRESHOLD}. Generating submission...")
        # Generate submission for the full test set
        generate_submission(load_cached_data=True, debug_size=None)
    else:
        print(f"\nValidation AUC does not exceed {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
