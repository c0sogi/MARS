import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_dataloader
from library.model import AsymmetricEfficientNet


def predict(model, loader, device):
    """
    Performs inference on the provided loader using the model.
    Applies Test-Time Augmentation (TTA): Original, Horizontal Flip, Vertical Flip.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): The data loader for the test set.
        device (torch.device): The compute device.

    Returns:
        np.ndarray: Array of predicted probabilities.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)

            # TTA 1: Original
            outputs_orig = model(inputs)
            probs_orig = torch.sigmoid(outputs_orig)

            # TTA 2: Horizontal Flip (dim 3 is Width in B, C, H, W)
            inputs_hflip = torch.flip(inputs, dims=[3])
            outputs_hflip = model(inputs_hflip)
            probs_hflip = torch.sigmoid(outputs_hflip)

            # TTA 3: Vertical Flip (dim 2 is Height in B, C, H, W)
            inputs_vflip = torch.flip(inputs, dims=[2])
            outputs_vflip = model(inputs_vflip)
            probs_vflip = torch.sigmoid(outputs_vflip)

            # Average predictions
            avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0

            all_probs.append(avg_probs.cpu().numpy())

    # Concatenate all batches
    if len(all_probs) > 0:
        return np.concatenate(all_probs).flatten()
    else:
        return np.array([])


def generate_submission(debug=False):
    """
    Main function to generate the submission file.
    Loads the model, processes the test set, runs inference with TTA, and saves the CSV.

    Args:
        debug (bool): If True, runs on a subset of data for debugging purposes.
    """
    # 1. Reproducibility
    set_seed(Config.SEED)

    # 2. Setup Device
    device = torch.device(Config.DEVICE)

    # 3. Load Test Metadata
    # We need this to map the predictions back to BraTS21IDs
    test_df = pd.read_csv(Config.TEST_METADATA)
    if debug:
        test_df = test_df.head(Config.DEBUG_SIZE)

    # 4. Data Loading
    # get_dataloader handles the complex preprocessing and caching logic
    test_loader = get_dataloader(
        split="test", batch_size=Config.BATCH_SIZE, debug=debug
    )

    # 5. Model Initialization
    model = AsymmetricEfficientNet(
        pretrained=False
    )  # Pretrained weights not needed for inference, we load our own

    # Load trained weights
    if os.path.exists(Config.MODEL_PATH):
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # Fallback for when running locally without a trained model (e.g. initial test)
        # In a real submission, this path must exist.
        print(
            f"Warning: Model checkpoint not found at {Config.MODEL_PATH}. Using random weights."
        )

    model = model.to(device)

    # 6. Inference
    predictions = predict(model, test_loader, device)

    # 7. Create Submission DataFrame
    # Ensure lengths match (loader might drop last batch if drop_last=True, but standard loader doesn't)
    # The get_dataloader processes the dataframe sequentially, so order is preserved.

    if len(predictions) != len(test_df):
        print(
            f"Warning: Mismatch between predictions ({len(predictions)}) and metadata ({len(test_df)})."
        )
        # Handle potential mismatch if debug/caching caused issues, though unlikely with current logic
        min_len = min(len(predictions), len(test_df))
        predictions = predictions[:min_len]
        test_df = test_df.iloc[:min_len]

    submission_df = pd.DataFrame(
        {"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": predictions}
    )

    # 8. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
