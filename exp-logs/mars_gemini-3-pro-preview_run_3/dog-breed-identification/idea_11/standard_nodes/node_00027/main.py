import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from PIL import Image

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, create_model_soup
from library.dataset import DogDataset, get_transforms, get_class_mapping
from library.model import get_model
from library.engine import train_fold, predict_with_tta
from torch.utils.data import DataLoader

# ==========================================
# Runtime Configuration Overrides
# ==========================================
# Adjust settings to ensure execution within time limits while retaining strategy
Config.EPOCHS = 15
Config.SOUP_EPOCH_START = 10
Config.N_FOLDS = 5
# Ensure output directory exists
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


def get_fixed_val_loader(classes, class_to_idx):
    """Creates a DataLoader for the fixed hold-out validation set."""
    df_val = pd.read_csv(Config.VAL_CSV)

    # Use the validation transform
    transform = get_transforms("val")

    dataset = DogDataset(
        df_val, transform=transform, class_to_idx=class_to_idx, is_test=False
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return loader, df_val


def get_test_loader_internal():
    """Creates a DataLoader for the test set."""
    df_test = pd.read_csv(Config.TEST_CSV)
    transform = get_transforms("test")

    dataset = DogDataset(df_test, transform=transform, is_test=True)

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return loader, df_test


def perform_failure_analysis(df_val, probs, true_labels, classes):
    """
    Analyzes model failures by correlating error magnitude with image metadata.
    """
    print("\n--- Failure Analysis ---")

    # 1. Calculate Per-Sample Log Loss
    # Extract probability of the true class
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)

    # Get the probability assigned to the true class for each sample
    # true_labels are indices
    true_class_probs = probs[np.arange(len(probs)), true_labels]
    sample_losses = -np.log(true_class_probs)

    # 2. Extract Metadata Features
    file_sizes = []
    aspect_ratios = []
    widths = []
    heights = []

    for idx, row in df_val.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # Get file size
            file_sizes.append(os.path.getsize(path))

            # Get dimensions
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 0)
        except Exception:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    # 3. Compute Correlations
    analysis_df = pd.DataFrame(
        {
            "loss": sample_losses,
            "file_size": file_sizes,
            "aspect_ratio": aspect_ratios,
            "width": widths,
            "height": heights,
        }
    )

    correlations = analysis_df.corr()["loss"].drop("loss")
    print("Correlation between Error Magnitude (Log Loss) and Input Features:")
    print(correlations)

    return analysis_df


def main():
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Get Class Mapping
    classes, class_to_idx = get_class_mapping()
    num_classes = len(classes)

    # 2. Train Folds and Create Soup Models
    fold_model_paths = []

    for fold_idx in range(Config.N_FOLDS):
        # Train the fold (Warmup + Fine-tune)
        # Returns list of checkpoint paths for the soup epochs
        soup_checkpoints = train_fold(fold_idx)

        # Create Model Soup
        soup_state_dict = create_model_soup(soup_checkpoints, device=device)

        # Save the consolidated soup model for this fold
        save_path = os.path.join(Config.OUTPUT_DIR, f"best_model_fold_{fold_idx}.pth")
        torch.save(soup_state_dict, save_path)
        fold_model_paths.append(save_path)

        # Clean up individual epoch checkpoints to save space (optional, skipping for safety)
        # Force garbage collection
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    # 3. Ensemble Inference
    print("\n--- Starting Ensemble Inference ---")

    # Load Validation and Test Loaders
    val_loader, df_val = get_fixed_val_loader(classes, class_to_idx)
    test_loader, df_test = get_test_loader_internal()

    # Placeholders for ensemble predictions
    val_probs_ensemble = np.zeros((len(df_val), num_classes))
    test_probs_ensemble = np.zeros((len(df_test), num_classes))

    # Iterate through each fold model
    for fold_idx, model_path in enumerate(fold_model_paths):
        print(f"Inference with Fold {fold_idx} Model Soup...")

        # Load Model
        model = get_model(
            device=device, pretrained=False
        )  # Pretrained=False as we load custom weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Predict on Validation (using TTA)
        val_probs = predict_with_tta(model, val_loader, device)
        val_probs_ensemble += val_probs

        # Predict on Test (using TTA)
        test_probs = predict_with_tta(model, test_loader, device)
        test_probs_ensemble += test_probs

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Average predictions (Bagging)
    val_probs_ensemble /= Config.N_FOLDS
    test_probs_ensemble /= Config.N_FOLDS

    # 4. Validation Metric
    # Get true labels for validation set
    y_true = [class_to_idx[breed] for breed in df_val["breed"]]

    final_metric = log_loss(y_true, val_probs_ensemble, labels=list(range(num_classes)))
    print(f"Final Validation Metric: {final_metric:.16f}")

    # 5. Failure Analysis
    perform_failure_analysis(df_val, val_probs_ensemble, np.array(y_true), classes)

    # 6. Submission
    # Threshold check
    THRESHOLD = 0.14004325100369866

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) meets threshold ({THRESHOLD:.6f}). Generating submission..."
        )

        # Create submission DataFrame
        # Format: id, breed1, breed2, ...
        submission_df = pd.DataFrame(test_probs_ensemble, columns=classes)
        submission_df.insert(0, "id", df_test["id"])

        # Save
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nMetric ({final_metric:.6f}) did NOT meet threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
