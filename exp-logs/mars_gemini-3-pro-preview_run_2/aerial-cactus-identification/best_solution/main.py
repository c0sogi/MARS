import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure library modules can be imported from the current directory
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import CactusResNet, train_model, predict_and_submit
from library.utils import load_checkpoint, calculate_auc


def main():
    # 1. Setup System
    Config.create_directories()
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing task on device: {device}")

    # 2. Prepare Data
    # Load data with caching enabled for efficiency
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Train Multiple Models (Seed Averaging)
    # Cite solution_lesson_node_00005: Using robust training strategy (Ensembling)
    print("Starting training with Seed Averaging...")

    for i in range(Config.NUM_SEEDS):
        print(f"\n--- Training Model Seed {i+1}/{Config.NUM_SEEDS} ---")

        # Initialize Model
        model = CactusResNet().to(device)

        # Define save path for this seed
        save_path = os.path.join(Config.WORKING_DIR, f"model_seed_{i}.pth")

        # Train Model
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=Config.NUM_EPOCHS,
            patience=Config.EARLY_STOPPING_PATIENCE,
            save_path=save_path,
            seed_val=Config.SEED + i,
        )

    # 5. Validation Assessment (Ensemble)
    print("Running inference on validation set (Ensemble)...")

    val_probs_ensemble = None
    val_targets = []

    # Storage for image stats for failure analysis (only needed once)
    val_stats = {
        "brightness": [],
        "contrast": [],
        "red_mean": [],
        "green_mean": [],
        "blue_mean": [],
    }
    stats_collected = False

    for i in range(Config.NUM_SEEDS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{i}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model for seed {i} not found at {model_path}")
            continue

        # Load model
        model = CactusResNet().to(device)
        load_checkpoint(model_path, model, device=device)
        model.eval()

        current_probs = []
        current_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)

                # Forward pass
                outputs = model(images).squeeze(1)
                probs = torch.sigmoid(outputs)

                current_probs.extend(probs.cpu().numpy())

                if i == 0:
                    current_targets.extend(labels.numpy())

                    # Extract features for failure analysis
                    if not stats_collected:
                        b_batch = images.mean(dim=[1, 2, 3]).cpu().numpy()
                        val_stats["brightness"].extend(b_batch)
                        c_batch = images.std(dim=[1, 2, 3]).cpu().numpy()
                        val_stats["contrast"].extend(c_batch)
                        means = images.mean(dim=[2, 3]).cpu().numpy()
                        val_stats["red_mean"].extend(means[:, 0])
                        val_stats["green_mean"].extend(means[:, 1])
                        val_stats["blue_mean"].extend(means[:, 2])

        if i == 0:
            val_targets = current_targets
            val_probs_ensemble = np.array(current_probs)
            stats_collected = True
        else:
            val_probs_ensemble += np.array(current_probs)

    # Average probabilities
    val_probs_ensemble /= Config.NUM_SEEDS

    # Calculate Metric
    val_auc = calculate_auc(val_targets, val_probs_ensemble)

    # Print Metric in required format
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    val_targets_arr = np.array(val_targets)

    # Calculate error magnitude
    errors = np.abs(val_targets_arr - val_probs_ensemble)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(val_stats)
    df_analysis["error"] = errors

    print("Correlation between Error Magnitude and Input Features:")
    # Calculate Pearson correlation
    correlations = df_analysis.corr()["error"].drop("error")
    print(correlations)

    # 7. Submission
    THRESHOLD = 0.9997903583412834

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Ensemble Prediction on Test Set
        print("Starting inference with Test Time Augmentation (Ensemble)...")
        test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
        test_ids = test_meta["id"].values

        test_probs_ensemble = None

        for i in range(Config.NUM_SEEDS):
            model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{i}.pth")
            model = CactusResNet().to(device)
            load_checkpoint(model_path, model, device=device)
            model.eval()

            current_test_probs = []

            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device)

                    # TTA: Original
                    out_orig = model(images).squeeze(1)
                    prob_orig = torch.sigmoid(out_orig)

                    # TTA: Horizontal
                    images_h = torch.flip(images, [3])
                    out_h = model(images_h).squeeze(1)
                    prob_h = torch.sigmoid(out_h)

                    # TTA: Vertical
                    images_v = torch.flip(images, [2])
                    out_v = model(images_v).squeeze(1)
                    prob_v = torch.sigmoid(out_v)

                    avg_prob = (prob_orig + prob_h + prob_v) / 3.0
                    current_test_probs.extend(avg_prob.cpu().numpy())

            if i == 0:
                test_probs_ensemble = np.array(current_test_probs)
            else:
                test_probs_ensemble += np.array(current_test_probs)

        test_probs_ensemble /= Config.NUM_SEEDS

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {"id": test_ids, "has_cactus": test_probs_ensemble}
        )
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({val_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
