import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import load_checkpoint, seed_everything
from library.models import FractureClassifier
from library.dataset import process_slice_metadata, FractureCropDataset
from library.segmentation_engine import generate_spine_coordinates
from library.classification_engine import inference_and_submission


def predict_study(study_uid, mode="test", device=Config.DEVICE):
    """
    Loads a scan, utilizes the SpineLocalizer to find ROIs, feeds crops to the
    FractureClassifier, and applies Max Pooling across slice predictions to
    generate the final patient-level and vertebrae-level probabilities.

    Args:
        study_uid (str): The StudyInstanceUID to predict.
        mode (str): 'train', 'val', or 'test' to locate images.
        device (str): Computation device.

    Returns:
        dict: Dictionary containing probabilities for C1-C7 and patient_overall.
    """
    seed_everything(Config.SEED)

    # 1. Setup Metadata for this single study
    # Create a 1-row DataFrame to be compatible with library functions
    meta_df = pd.DataFrame({"StudyInstanceUID": [study_uid]})

    # 2. Process Slices
    # We disable caching to prevent overwriting the full dataset cache with this single study
    # Note: process_slice_metadata relies on 'mode' to select the correct image directory
    slice_df = process_slice_metadata(
        meta_df, bbox_df=None, mode=mode, load_cached_data=False
    )

    if len(slice_df) == 0:
        # Fallback if no slices are found
        return {col: 0.0 for col in Config.TARGET_COLS}

    # 3. Generate Coordinates (Localizer)
    # This runs the SpineLocalizer inference for this specific study to find ROIs
    coords_map = generate_spine_coordinates(
        meta_df, mode=mode, load_cached_data=False, batch_size=Config.BATCH_SIZE
    )

    # 4. Prepare Dataset for Classifier
    # Uses the generated coordinates to create 2.5D crops
    dataset = FractureCropDataset(slice_df, coords_map=coords_map, mode=mode)

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Load Classifier Model
    model = FractureClassifier(pretrained=False).to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "fracture_classifier.pth")

    # Load weights if available
    if os.path.exists(checkpoint_path):
        model, _, _, _ = load_checkpoint(model, None, checkpoint_path, device=device)

    model.eval()

    # 6. Inference Loop
    all_probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())

    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs, axis=0)
    else:
        return {col: 0.0 for col in Config.TARGET_COLS}

    # 7. Aggregation (Max Pooling)
    # all_probs shape: (num_slices, 8)
    # We take the maximum probability across all slices for each class
    study_max_probs = np.max(all_probs, axis=0)

    results = {}
    for idx, col in enumerate(Config.TARGET_COLS):
        results[col] = float(study_max_probs[idx])

    return results


def generate_submission(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Generates predictions for the entire test set and creates the submission file.

    This function leverages the optimized batch processing pipeline from the
    library's classification engine to ensure efficiency and correctness.

    Args:
        batch_size (int): Batch size for inference.
        load_cached_data (bool): Whether to use cached metadata/coordinates.
    """
    inference_and_submission(batch_size=batch_size, load_cached_data=load_cached_data)
