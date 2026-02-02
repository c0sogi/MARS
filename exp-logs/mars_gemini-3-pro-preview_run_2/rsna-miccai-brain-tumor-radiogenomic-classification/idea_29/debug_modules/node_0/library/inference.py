import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import (
    TEST_METADATA_PATH,
    TEST_CACHE_PATH,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    SEED,
)
from library.dataset import load_dataset
from library.model import AsymmetricEfficientNet


def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict_batch_with_tta(model, inputs, device):
    """
    Predicts probabilities for a batch of inputs using Test-Time Augmentation (TTA).
    Applies Horizontal and Vertical flips.

    Args:
        model (nn.Module): The trained model.
        inputs (torch.Tensor): Batch of input tensors (B, C, H, W).
        device (str): Computation device.

    Returns:
        np.ndarray: Averaged probabilities for the batch.
    """
    inputs = inputs.to(device)

    # 1. Original
    logits_orig = model(inputs)
    probs_orig = torch.sigmoid(logits_orig)

    # 2. Horizontal Flip (dim 3 is width)
    inputs_h = torch.flip(inputs, dims=[3])
    logits_h = model(inputs_h)
    probs_h = torch.sigmoid(logits_h)

    # 3. Vertical Flip (dim 2 is height)
    inputs_v = torch.flip(inputs, dims=[2])
    logits_v = model(inputs_v)
    probs_v = torch.sigmoid(logits_v)

    # Average probabilities
    avg_probs = (probs_orig + probs_h + probs_v) / 3.0

    return avg_probs.detach().cpu().numpy().flatten()


def generate_submission(debug_max_samples=None, load_cached_data=True):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        debug_max_samples (int, optional): Limit number of samples for debugging.
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    set_seed()
    print("Starting Inference Generation...")

    # 1. Load Test Data
    # load_dataset handles caching and raw DICOM processing if cache is missing
    test_dataset = load_dataset(
        metadata_path=TEST_METADATA_PATH,
        cache_path_data=TEST_CACHE_PATH,
        cache_path_labels=None,  # No labels for test set
        load_cached_data=load_cached_data,
        transform=None,  # TTA is handled manually in the loop
        debug_max_samples=debug_max_samples,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Model
    model = AsymmetricEfficientNet()
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading model weights from {MODEL_SAVE_PATH}")
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Model file not found at {MODEL_SAVE_PATH}. Using random weights."
        )

    model.to(DEVICE)
    model.eval()

    # 3. Inference Loop
    all_preds = []

    print("Running prediction loop with TTA...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            # inputs shape: (B, 12, 224, 224)
            # labels are dummy (-1) and ignored
            batch_preds = predict_batch_with_tta(model, inputs, DEVICE)
            all_preds.extend(batch_preds)

    # 4. Create Submission DataFrame
    # Reload metadata to ensure ID alignment
    df_test = pd.read_csv(TEST_METADATA_PATH)
    if debug_max_samples is not None:
        df_test = df_test.head(debug_max_samples)

    # Ensure length match
    if len(all_preds) != len(df_test):
        print(
            f"Warning: Prediction count ({len(all_preds)}) does not match Metadata count ({len(df_test)}). Truncating/Padding."
        )
        # This theoretically shouldn't happen if dataset loading is consistent
        if len(all_preds) > len(df_test):
            all_preds = all_preds[: len(df_test)]
        else:
            all_preds.extend([0.5] * (len(df_test) - len(all_preds)))

    submission = pd.DataFrame(
        {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": all_preds}
    )

    # 5. Save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission.head())
