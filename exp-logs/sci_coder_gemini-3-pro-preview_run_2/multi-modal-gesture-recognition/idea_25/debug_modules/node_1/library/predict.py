import os
import torch
import numpy as np
import scipy.ndimage
from torch.utils.data import DataLoader

from library.config import (
    WORKING_DIR,
    TEST_METADATA_PATH,
    SUBMISSION_FILE,
    BATCH_SIZE,
    SEED,
)
from library.utils import set_seed, load_checkpoint
from library.data_loader import GestureDataset, collate_fn
from library.model import GSG_CRCN


def post_process_sequence(pred_indices, kernel_size=15):
    """
    Applies median filtering with nearest-neighbor padding to smooth predictions,
    then collapses repeated labels and removes the background class.

    Args:
        pred_indices (np.array): 1D array of frame-wise class indices.
        kernel_size (int): Size of the median filter kernel.

    Returns:
        list[int]: Ordered list of recognized gesture IDs.
    """
    if len(pred_indices) == 0:
        return []

    # 1. Median Filter with Nearest-Neighbor Padding
    # mode='nearest' repeats the edge values, preventing boundary artifacts
    filtered = scipy.ndimage.median_filter(
        pred_indices, size=kernel_size, mode="nearest"
    )

    # 2. Collapse Repeats and Remove Background (0)
    final_seq = []
    prev = -1
    for label in filtered:
        if label != prev:
            if label != 0:
                final_seq.append(int(label))
            prev = label

    return final_seq


def generate_predictions(
    checkpoint_path=None,
    output_file=SUBMISSION_FILE,
    subset_size=None,
    batch_size=BATCH_SIZE,
    median_kernel_size=15,
):
    """
    Runs the inference pipeline on the test dataset and generates the submission CSV.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint. Defaults to best_model.pth.
        output_file (str): Path to save the submission file.
        subset_size (int, optional): Number of samples to process (for debugging).
        batch_size (int): Batch size for inference.
        median_kernel_size (int): Kernel size for the post-processing median filter.
    """
    # Ensure reproducibility
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")

    # 1. Load Test Data
    # GestureDataset handles caching automatically in ./working/idea_25/
    print("Loading Test Dataset...")
    test_dataset = GestureDataset(
        TEST_METADATA_PATH, is_train=False, augment=False, subset_size=subset_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Load Model
    print("Loading Model...")
    model = GSG_CRCN().to(device)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(WORKING_DIR, "best_model.pth")

    try:
        load_checkpoint(checkpoint_path, model)
        print(f"Loaded model weights from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()

    # 3. Run Inference
    print("Running Inference...")
    results = []

    with torch.no_grad():
        for batch_data in test_loader:
            features, _, _, lengths, mask, sample_ids = batch_data

            features = features.to(device)
            mask = mask.to(device)

            # Forward Pass
            outputs = model(features, mask)

            # Extract Stage 3 Class Logits
            # Model output structure: outputs['stage3']['cls'] -> (N, L, C)
            logits = outputs["stage3"]["cls"]

            # Get discrete predictions
            preds = torch.argmax(logits, dim=2).cpu().numpy()  # (N, L)

            # Process each sequence in the batch
            for i, sample_id in enumerate(sample_ids):
                length = lengths[i]

                # Slice to valid sequence length
                raw_seq = preds[i, :length]

                # Post-process (Filter -> Collapse -> Clean)
                final_seq = post_process_sequence(
                    raw_seq, kernel_size=median_kernel_size
                )

                # Format as comma-separated string
                pred_str = ",".join(map(str, final_seq))
                results.append((sample_id, pred_str))

    # 4. Save Submission
    print(f"Saving {len(results)} predictions to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        for sample_id, pred_str in results:
            # Format: SessionID,Label1,Label2,...
            f.write(f"{sample_id},{pred_str}\n")

    print("Inference Complete.")
