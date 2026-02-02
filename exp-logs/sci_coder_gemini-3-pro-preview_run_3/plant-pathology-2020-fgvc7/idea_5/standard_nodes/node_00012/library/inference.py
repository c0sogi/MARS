import os
import torch
import pandas as pd
import numpy as np

from library.config import Config
from library.dataset import get_test_dataloader
from library.models import MultiLevelEfficientNet, SwinTransformerModel


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation.
    Performs inference on Original, Horizontally Flipped, and Vertically Flipped images.

    Args:
        model: The loaded neural network.
        loader: DataLoader for the test set.
        device: Computation device.

    Returns:
        np.ndarray: Averaged probability predictions of shape (N_samples, N_classes).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)

            # 1. Forward pass on original images
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            if Config.USE_TTA:
                # 2. Forward pass on Horizontally Flipped images
                # Tensor shape is [B, C, H, W]. Dim 3 is Width.
                images_h = torch.flip(images, dims=[3])
                outputs_h = model(images_h)
                probs_h = torch.softmax(outputs_h, dim=1)

                # 3. Forward pass on Vertically Flipped images
                # Tensor shape is [B, C, H, W]. Dim 2 is Height.
                images_v = torch.flip(images, dims=[2])
                outputs_v = model(images_v)
                probs_v = torch.softmax(outputs_v, dim=1)

                # 4. Forward pass on Transposed images
                # Tensor shape is [B, C, H, W]. Swap H and W.
                images_t = torch.transpose(images, 2, 3)
                outputs_t = model(images_t)
                probs_t = torch.softmax(outputs_t, dim=1)

                # Average the probabilities
                probs = (probs + probs_h + probs_v + probs_t) / 4.0

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def generate_submission(debug: bool = False):
    """
    Main inference routine. Loads all trained models, performs TTA inference,
    ensembles predictions, and saves the submission file.

    Args:
        debug (bool): If True, runs inference on a small subset of the test data.
    """
    device = Config.DEVICE

    # 1. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_METADATA}")

    test_df = pd.read_csv(Config.TEST_METADATA)

    if debug:
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        print(f"Debug mode: Inference on {len(test_df)} samples.")

    # Initialize accumulator for ensemble predictions
    # Shape: (N_samples, N_classes)
    ensemble_probs = np.zeros((len(test_df), Config.NUM_CLASSES), dtype=np.float64)
    model_count = 0

    # 2. Define Models to Ensemble
    # We iterate over both architectures and all folds
    architectures = [("effnet", Config.IMG_SIZE_EFFNET), ("swin", Config.IMG_SIZE_SWIN)]

    print("Starting Inference Ensemble...")

    for model_type, img_size in architectures:
        # Create DataLoader for this specific image size
        # We create it once per architecture to avoid overhead
        loader = get_test_dataloader(test_df, img_size, Config.BATCH_SIZE)

        for fold in range(Config.N_FOLDS):
            # Construct checkpoint path
            checkpoint_path = os.path.join(
                Config.WORKING_DIR, f"{model_type}_fold_{fold}_best.pth"
            )

            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found: {checkpoint_path}. Skipping.")
                continue

            print(f"Loading {model_type} (Fold {fold})...")

            # Initialize Model Architecture
            # We set pretrained=False because we are loading custom weights
            if model_type == "effnet":
                model = MultiLevelEfficientNet(pretrained=False)
            elif model_type == "swin":
                model = SwinTransformerModel(pretrained=False)
            else:
                continue

            # Load Weights
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)

            # Generate Predictions
            preds = predict_with_tta(model, loader, device)

            # Accumulate
            ensemble_probs += preds
            model_count += 1

            # Clean up to save memory
            del model
            torch.cuda.empty_cache()

    if model_count == 0:
        raise RuntimeError("No models were loaded. Cannot generate submission.")

    print(f"Inference complete. Ensembled {model_count} models.")

    # 3. Average Predictions
    final_probs = ensemble_probs / model_count

    # 4. Create Submission DataFrame
    submission_df = pd.DataFrame(final_probs, columns=Config.CLASSES)

    # Insert image_id at the beginning
    submission_df.insert(0, "image_id", test_df["image_id"])

    # 5. Save Submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
