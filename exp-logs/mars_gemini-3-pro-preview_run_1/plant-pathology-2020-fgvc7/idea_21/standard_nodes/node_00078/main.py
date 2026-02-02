import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config, seed_everything
from library.utils import get_device
from library.dataset import AppleDataset, get_valid_transforms
from library.model import AppleResNet34
from library.train_eval import train_single_fold, generate_submission


def evaluate_ensemble(val_metadata_path, model_paths, device):
    """
    Evaluates the ensemble of trained models on the fixed hold-out validation set.
    """
    print(f"\nEvaluating Ensemble on {val_metadata_path}...")

    # Load Metadata
    df_val = pd.read_csv(val_metadata_path)

    # Create Dataset and Loader
    val_dataset = AppleDataset(
        df=df_val, transforms=get_valid_transforms(), is_test=False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    ensemble_probs = None
    all_targets = []

    # Iterate through each trained model in the ensemble
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Skipping.")
            continue

        # Initialize model and load weights
        model = AppleResNet34(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        fold_probs = []
        fold_targets = []

        # Inference loop
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)

                logits = model(images)
                probs = torch.softmax(logits, dim=1)

                fold_probs.append(probs.cpu().numpy())
                fold_targets.append(targets.numpy())

        fold_probs = np.vstack(fold_probs)

        # Accumulate probabilities for averaging
        if ensemble_probs is None:
            ensemble_probs = fold_probs
            all_targets = np.vstack(fold_targets)
        else:
            ensemble_probs += fold_probs

    if ensemble_probs is None:
        print("Error: No models were evaluated.")
        return 0.0, None, None

    # Compute average probabilities
    avg_probs = ensemble_probs / len(model_paths)

    # Calculate Mean Column-wise ROC AUC
    try:
        auc = roc_auc_score(all_targets, avg_probs, average="macro")
    except Exception as e:
        print(f"Metric calculation failed: {e}")
        auc = 0.0

    return auc, all_targets, avg_probs


def failure_analysis(df_val, targets, preds):
    """
    Performs failure analysis by correlating error magnitude with image meta-features.
    """
    print("\nPerforming Failure Analysis...")

    # Calculate Mean Absolute Error per sample
    errors = np.mean(np.abs(targets - preds), axis=1)

    # Extract Meta-Features from images
    widths = []
    heights = []
    intensities = []

    for idx, row in df_val.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        if os.path.exists(full_path):
            img = cv2.imread(full_path)
            if img is not None:
                h, w, c = img.shape
                widths.append(w)
                heights.append(h)
                # Calculate normalized mean intensity
                intensities.append(img.mean() / 255.0)
            else:
                widths.append(0)
                heights.append(0)
                intensities.append(0)
        else:
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(
        {"error": errors, "width": widths, "height": heights, "intensity": intensities}
    )

    # Calculate Correlations
    print("Correlation between Error and Meta-features:")
    correlations = analysis_df.corrwith(analysis_df["error"])
    for col in ["width", "height", "intensity"]:
        print(f"  {col}: {correlations[col]:.4f}")


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for optimized execution within time limits
    Config.EPOCHS = 8
    Config.N_SPLITS = 5
    Config.BATCH_SIZE = 64  # Utilize A100 memory

    seed_everything(42)
    device = get_device()

    print(f"Starting execution for {Config.IDEA_NAME}")
    print(
        f"Configuration: {Config.N_SPLITS} Splits, {Config.EPOCHS} Epochs, Batch Size {Config.BATCH_SIZE}"
    )

    # Ensure working directories exist
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # 2. Training Pipeline
    # ==========================================
    trained_model_paths = []

    # Use the configured number of seeds/splits
    seeds = Config.SEEDS[: Config.N_SPLITS]

    for i, seed in enumerate(seeds):
        # Train model for this split
        # train_single_fold handles data splitting, model init, training, and saving
        train_single_fold(split_seed=seed, fold_idx=i)

        # Record the path of the saved model
        model_path = os.path.join(Config.MODEL_DIR, f"resnet34_seed_{seed}.pth")
        trained_model_paths.append(model_path)

    # ==========================================
    # 3. Ensemble Validation
    # ==========================================
    val_auc, val_targets, val_preds = evaluate_ensemble(
        Config.VAL_METADATA_PATH, trained_model_paths, device
    )

    # Print required metric
    print(f"Final Validation Metric: {val_auc:.16f}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    failure_analysis(df_val, val_targets, val_preds)

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    threshold = 0.9901680711448418

    if val_auc > threshold:
        print(
            f"\nValidation metric {val_auc:.6f} exceeds threshold {threshold:.6f}. Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nValidation metric {val_auc:.6f} does not exceed threshold {threshold:.6f}. Submission skipped."
        )


if __name__ == "__main__":
    main()
