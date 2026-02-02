import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.data import get_dataloaders
from library.model import create_model


def predict_with_tta(model, loader, device):
    """
    Performs inference on the data loader using the provided model.
    Applies Test Time Augmentation (Horizontal Flip) if configured.

    Args:
        model: PyTorch model in eval mode.
        loader: DataLoader for the test set.
        device: Computation device (cpu/cuda).

    Returns:
        np.ndarray: Softmax probabilities of shape (N, num_classes).
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)

            # 1. Standard View Forward Pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # 2. TTA: Horizontal Flip
            if Config.USE_TTA:
                # Flip the width dimension (dim 3 for NCHW tensor)
                images_flipped = torch.flip(images, dims=[3])
                outputs_flipped = model(images_flipped)
                probs_flipped = torch.softmax(outputs_flipped, dim=1)

                # Average predictions
                probs = (probs + probs_flipped) / 2.0

            all_probs.append(probs.cpu())

    # Concatenate all batches
    return torch.cat(all_probs, dim=0).numpy()


def ensemble_predictions(models, loader, device):
    """
    Generates ensemble predictions by averaging probabilities from multiple models.

    Args:
        models: List of loaded PyTorch models.
        loader: DataLoader for the test set.
        device: Computation device.

    Returns:
        np.ndarray: Averaged softmax probabilities.
    """
    avg_probs = None

    for i, model in enumerate(models):
        # Get probabilities for this model (includes TTA if enabled)
        probs = predict_with_tta(model, loader, device)

        if avg_probs is None:
            avg_probs = probs
        else:
            avg_probs += probs

    # Average over the number of models
    if avg_probs is not None:
        avg_probs /= len(models)

    return avg_probs


def run_inference(
    checkpoint_dir=Config.CHECKPOINT_DIR,
    output_dir=Config.SUBMISSION_DIR,
    load_cached_data=False,
    debug=Config.DEBUG,
    debug_size=Config.DEBUG_SUBSET_SIZE,
):
    """
    Main inference pipeline. Loads models, generates predictions, and saves submission.

    Args:
        checkpoint_dir: Directory containing model checkpoints.
        output_dir: Directory to save the submission file.
        load_cached_data: If True, attempts to load existing submission.
        debug: If True, runs on a small subset of data.
        debug_size: Number of samples to use in debug mode.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Check for cached submission
    submission_path = os.path.join(output_dir, "submission.csv")
    if load_cached_data and os.path.exists(submission_path):
        print(f"Loading cached submission from {submission_path}")
        return pd.read_csv(submission_path)

    # 1. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_METADATA}")

    test_df = pd.read_csv(Config.TEST_METADATA)

    # Handle Debugging
    if debug:
        print(f"Debug mode enabled. Using first {debug_size} samples.")
        test_df = test_df.iloc[:debug_size].reset_index(drop=True)

    print(f"Total test samples: {len(test_df)}")

    # 2. Prepare DataLoader
    # We use Phase 2 configuration (higher resolution) for inference
    phase_config = Config.PHASE_2

    # Create dummy DataFrames for train/val as get_dataloaders requires them
    dummy_df = pd.DataFrame(columns=["image_id", "label", "file_path"])

    # We only care about the test_loader
    _, _, test_loader = get_dataloaders(dummy_df, dummy_df, test_df, phase_config)

    # 3. Load Models (Ensemble)
    models = []
    print(f"Loading models from {checkpoint_dir}...")

    for fold_idx in range(Config.NUM_FOLDS):
        # Check for best model file for this fold
        checkpoint_path = os.path.join(
            checkpoint_dir, f"best_model_fold_{fold_idx}.pth"
        )

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint for fold {fold_idx} not found. Skipping.")
            continue

        # Initialize model architecture
        # We don't need to download pretrained weights since we are loading a checkpoint
        model = create_model(pretrained=False)
        model.to(device)

        # Load weights
        try:
            model, _ = load_checkpoint(model, checkpoint_path, device)
            model.eval()
            models.append(model)
            print(f"Loaded model for fold {fold_idx}")
        except Exception as e:
            print(f"Failed to load checkpoint for fold {fold_idx}: {e}")

    if not models:
        raise RuntimeError("No models were loaded. Cannot perform inference.")

    # 4. Generate Predictions
    print(
        f"Starting inference with {len(models)} models (TTA={'Enabled' if Config.USE_TTA else 'Disabled'})..."
    )
    final_probs = ensemble_predictions(models, test_loader, device)

    # 5. Create Submission File
    predictions = np.argmax(final_probs, axis=1)

    submission_df = pd.DataFrame(
        {"image_id": test_df["image_id"], "label": predictions}
    )

    # Save submission
    os.makedirs(output_dir, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    return submission_df
