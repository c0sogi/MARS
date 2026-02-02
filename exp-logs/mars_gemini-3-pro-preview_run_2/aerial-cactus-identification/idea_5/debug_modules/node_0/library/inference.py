import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import CactusDataset, get_test_transforms
from library.model import CustomSEResNet


def predict_tta(model, dataloader, device):
    """
    Performs Test Time Augmentation (TTA) inference on the provided model.

    Strategy:
    1. Predict on original image.
    2. Predict on horizontally flipped image.
    3. Predict on vertically flipped image.
    4. Average the probabilities (arithmetic mean).

    Args:
        model (nn.Module): The trained PyTorch model.
        dataloader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Array of predicted probabilities with shape (N, 1).
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            # 1. Original View
            outputs_orig = model(inputs)
            probs_orig = torch.sigmoid(outputs_orig)

            # 2. Horizontal Flip View
            inputs_h = torch.flip(inputs, dims=[3])
            outputs_h = model(inputs_h)
            probs_h = torch.sigmoid(outputs_h)

            # 3. Vertical Flip View
            inputs_v = torch.flip(inputs, dims=[2])
            outputs_v = model(inputs_v)
            probs_v = torch.sigmoid(outputs_v)

            # Average probabilities across views
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0
            all_probs.append(avg_probs.cpu().numpy())

    # Concatenate all batches along the 0-th dimension
    return np.concatenate(all_probs, axis=0)


def run_inference():
    """
    Main inference routine.
    - Loads the test dataset.
    - Performs Homogeneous Seed Averaging:
        - Loads models for all seeds in Config.SEEDS.
        - Runs TTA inference for each.
        - Averages the results.
    - Saves the final submission file.
    """
    device = torch.device(Config.DEVICE)
    print(f"Running inference on device: {device}")

    # 1. Prepare Test Data
    # Using mode='test' ensures we use the test metadata and cache
    test_dataset = CactusDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        transform=get_test_transforms(),
        mode="test",
        load_cached_data=True,
    )

    # Shuffle must be False to maintain alignment with dataset.ids
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test dataset loaded. Size: {len(test_dataset)} images.")

    # 2. Initialize Accumulator for Ensemble Predictions
    # Shape: (N, 1)
    ensemble_preds = np.zeros((len(test_dataset), 1), dtype=np.float32)
    successful_seeds = 0

    # 3. Iterate over Seeds (Homogeneous Ensemble)
    for seed in Config.SEEDS:
        model_path = Config.get_model_path(seed)

        if not os.path.exists(model_path):
            print(
                f"Warning: Checkpoint for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        print(f"Processing seed {seed}...")

        # Initialize model architecture
        model = CustomSEResNet(**Config.MODEL_PARAMS)

        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        # Get TTA predictions for this seed
        preds = predict_tta(model, test_loader, device)

        # Accumulate
        ensemble_preds += preds
        successful_seeds += 1

    if successful_seeds == 0:
        raise RuntimeError(
            "No valid model checkpoints found. Cannot generate submission."
        )

    # 4. Compute Final Average
    final_probs = ensemble_preds / successful_seeds

    # Flatten to 1D array for DataFrame
    final_probs = final_probs.flatten()

    # 5. Create Submission DataFrame
    # CactusDataset stores IDs in self.ids which aligns with the loader order
    submission_df = pd.DataFrame({"id": test_dataset.ids, "has_cactus": final_probs})

    # 6. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")
    print("First 5 predictions:")
    print(submission_df.head())
