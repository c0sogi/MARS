import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import scipy.ndimage

from library.config import (
    WORKING_DIR,
    SUBMISSION_DIR,
    TEST_METADATA_PATH,
    NUM_CLASSES,
    SEED,
    MEDIAN_FILTER_KERNEL,
)
from library.utils import set_seed
from library.model import SSG_CRCN
from library.data_loader import prepare_dataset, GestureDataset, collate_fn


def decode_sequence(frame_labels):
    """
    Decodes a frame-wise label sequence into a list of gesture IDs.
    Collapses repeats and removes background (class 0).

    Args:
        frame_labels (np.array): Array of frame-wise class indices.

    Returns:
        list: Ordered list of gesture IDs.
    """
    seq = []
    prev = -1
    for l in frame_labels:
        l = int(l)
        if l != prev:
            if l != 0:  # 0 is background
                seq.append(l)
            prev = l
    return seq


def post_process_sequence(preds):
    """
    Applies post-processing to the raw frame predictions.

    1. Median Filter with Nearest-Neighbor padding to smooth noise while preserving boundaries.
    2. Sequence decoding to collapse repeats and remove background.

    Args:
        preds (np.array): Raw frame-wise predictions.

    Returns:
        list: Decoded gesture sequence.
    """
    # 1. Median Filter
    if MEDIAN_FILTER_KERNEL > 1:
        # Use ndimage.median_filter with mode='nearest' to protect boundaries
        # unlike signal.medfilt which zero-pads
        preds = scipy.ndimage.median_filter(
            preds, size=MEDIAN_FILTER_KERNEL, mode="nearest"
        )

    # 2. Decode
    decoded_seq = decode_sequence(preds)
    return decoded_seq


def generate_predictions():
    """
    Main inference function.
    Loads the best model, processes the test dataset, runs inference,
    applies post-processing, and saves the submission file.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print(f"Error: Model file not found at {best_model_path}")
        return

    # 1. Load Test Data
    # prepare_dataset handles caching internally via load_cached_data=True
    print("Loading test data...")
    test_pos, test_aud, test_lbl, test_bnd, test_ids = prepare_dataset(
        TEST_METADATA_PATH, "test_data", load_cached_data=True
    )

    # 2. Create Dataset and Loader
    # Augmentation is disabled for inference
    test_dataset = GestureDataset(test_pos, test_aud, test_lbl, test_bnd, augment=False)

    # Use batch_size=1 to handle variable lengths cleanly during inference
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=1
    )

    # 3. Load Model
    print(f"Loading model from {best_model_path}...")
    model = SSG_CRCN().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    predictions = []

    print("Running inference...")
    with torch.no_grad():
        for i, (feats, _, _, mask) in enumerate(test_loader):
            feats = feats.to(device)
            mask = mask.to(device)

            # Forward pass
            outputs = model(feats, mask)

            # Use the output from the final stage (Stage 3)
            final_stage_out = outputs[-1]  # Shape: (B, C+1, T)

            # Extract class logits (first NUM_CLASSES channels)
            cls_logits = final_stage_out[:, :NUM_CLASSES, :]

            # Compute probabilities
            probs = F.softmax(cls_logits, dim=1)

            # Get predicted labels (Argmax)
            # Shape: (B, T) -> (1, T) -> (T,)
            preds = torch.argmax(probs, dim=1).cpu().numpy()[0]

            # Mask out padding to get the actual sequence length
            # mask is (B, T), here B=1
            valid_len = int(mask[0].sum().item())
            preds = preds[:valid_len]

            # Post-process (Filter + Decode)
            decoded_seq = post_process_sequence(preds)

            # Convert to comma-separated string format
            pred_str = ",".join(map(str, decoded_seq))
            predictions.append(pred_str)

    # 4. Write Submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    print(f"Saving submission to {submission_path}...")

    with open(submission_path, "w") as f:
        for sid, pred in zip(test_ids, predictions):
            f.write(f"{sid},{pred}\n")

    print("Submission generation complete.")
