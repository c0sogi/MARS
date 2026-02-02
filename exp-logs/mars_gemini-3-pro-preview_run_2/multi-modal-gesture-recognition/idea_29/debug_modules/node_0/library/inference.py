import os
import torch
import numpy as np
import scipy.ndimage
from torch.utils.data import DataLoader

from library import config, utils, data_loader, model


def post_process_predictions(frame_labels, filter_size=7):
    """
    Applies label-space smoothing and decodes the sequence.

    Args:
        frame_labels (np.ndarray): 1D array of frame-wise class indices.
        filter_size (int): Kernel size for the median filter.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(frame_labels) == 0:
        return []

    # Ensure filter_size is valid
    if filter_size % 2 == 0:
        filter_size += 1

    # Apply Median Filter with Nearest-Neighbor Padding
    # This preserves edges better than zero-padding
    if len(frame_labels) >= filter_size:
        smoothed_labels = scipy.ndimage.median_filter(
            frame_labels, size=filter_size, mode="nearest"
        )
    else:
        smoothed_labels = frame_labels

    # Decode: Collapse repeats and remove background
    sequence = []
    last_label = -1

    for lbl in smoothed_labels:
        lbl = int(lbl)
        if lbl != last_label:
            if lbl != 0:  # 0 is background
                sequence.append(lbl)
            last_label = lbl

    return sequence


def run_inference(
    checkpoint_path=os.path.join(config.WORKING_DIR, "best_model.pth"),
    output_path=config.SUBMISSION_FILE_PATH,
    batch_size=config.BATCH_SIZE,
    load_cached_data=True,
    device=config.DEVICE,
):
    """
    Runs inference on the test dataset and generates the submission file.

    Args:
        checkpoint_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        load_cached_data (bool): Whether to use cached dataset files.
        device (str): Device to run inference on.
    """
    # Set seeds for reproducibility
    utils.set_seed(config.SEED)

    print(f"Running inference on device: {device}")

    # 1. Load Test Data
    # The data_loader handles caching internally based on the split type detected in metadata path
    print("Loading test dataset...")
    test_dataset = data_loader.GestureDataset(
        config.TEST_METADATA_PATH, is_train=False, load_cached_data=load_cached_data
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=data_loader.collate_fn,
        pin_memory=True,
    )

    # 2. Load Model
    print(f"Loading model from {checkpoint_path}...")
    net = model.CASGCN().to(device)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights."
        )

    net.eval()

    predictions = {}

    # 3. Inference Loop
    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)
            sample_ids = batch["sample_ids"]

            # Forward pass
            outputs = net(features, mask, lengths)

            # Use Stage 3 output for final predictions
            # outputs['stage3'] is (cls_logits, bnd_logits)
            stage3_cls_logits = outputs["stage3"][0]  # (B, T, C)

            # Get hard predictions
            batch_preds = torch.argmax(stage3_cls_logits, dim=2).cpu().numpy()  # (B, T)
            batch_lengths = lengths.cpu().numpy()

            # Process each sample in the batch
            for i, sample_id in enumerate(sample_ids):
                valid_len = batch_lengths[i]

                # Extract valid frames
                raw_seq = batch_preds[i, :valid_len]

                # Post-process
                final_seq = post_process_predictions(raw_seq, filter_size=7)

                predictions[sample_id] = final_seq

    # 4. Save Submission
    print(f"Saving predictions for {len(predictions)} samples...")
    utils.save_submission(predictions, output_path)
    print("Inference complete.")
