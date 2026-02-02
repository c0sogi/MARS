import os
import torch
import numpy as np
import pandas as pd
import scipy.ndimage
from torch.utils.data import DataLoader
from itertools import groupby

from library.config import (
    DEVICE,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    TEST_METADATA_PATH,
    BATCH_SIZE,
    SEED,
    NUM_CLASSES,
)
from library.utils import set_seed
from library.data_loader import GestureDataset, collate_fn
from library.model import DCHGNet


def apply_median_filter(predictions, kernel_size=7):
    """
    Applies a median filter to smooth the sequence of predictions.
    Uses nearest-neighbor padding to handle boundaries.

    Args:
        predictions (np.ndarray): 1D array of class indices.
        kernel_size (int): Size of the median filter window.

    Returns:
        np.ndarray: Smoothed predictions.
    """
    # mode='nearest' replicates the edge values, satisfying the "Nearest-Neighbor Padding" requirement
    return scipy.ndimage.median_filter(predictions, size=kernel_size, mode="nearest")


def decode_sequence(predictions):
    """
    Decodes frame-wise predictions into a sequence of gesture IDs.
    1. Collapses consecutive duplicates.
    2. Removes background class (0).

    Args:
        predictions (np.ndarray): 1D array of class indices.

    Returns:
        list: Ordered list of gesture IDs (int).
    """
    # Collapse consecutive duplicates
    collapsed = [k for k, g in groupby(predictions)]

    # Remove background (class 0)
    # Gesture IDs are 1-20
    sequence = [int(x) for x in collapsed if x != 0]

    return sequence


class Predictor:
    def __init__(self, checkpoint_path, device):
        """
        Args:
            checkpoint_path (str): Path to the model checkpoint.
            device (torch.device): Device to run inference on.
        """
        self.device = device
        self.model = DCHGNet().to(device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        print(f"Loading model from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, dataloader):
        """
        Runs inference on the dataloader and returns decoded sequences.

        Args:
            dataloader (DataLoader): Validation or Test dataloader.

        Returns:
            list: List of decoded sequences (lists of ints).
        """
        all_sequences = []

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"]  # Keep on CPU for slicing

                # Forward pass
                outputs = self.model(features, mask)

                # Use Stage 3 Classification Logits
                # Shape: (B, T, C)
                logits = outputs["stage3_cls"]

                # Get class indices
                # Shape: (B, T)
                predictions = torch.argmax(logits, dim=2).cpu().numpy()

                # Process each sample in the batch
                for i in range(features.size(0)):
                    length = lengths[i].item()

                    # Slice valid frames
                    sample_pred = predictions[i, :length]

                    # 1. Smooth
                    smoothed_pred = apply_median_filter(sample_pred, kernel_size=7)

                    # 2. Decode
                    decoded_seq = decode_sequence(smoothed_pred)

                    all_sequences.append(decoded_seq)

        return all_sequences


def run_inference(
    checkpoint_name="best_model.pth",
    batch_size=BATCH_SIZE,
    load_cached_data=True,
    output_filename="submission.csv",
):
    """
    Main function to run inference and generate submission file.

    Args:
        checkpoint_name (str): Name of the checkpoint file in CHECKPOINT_DIR.
        batch_size (int): Batch size for inference.
        load_cached_data (bool): Whether to use cached preprocessed data.
        output_filename (str): Name of the output CSV file.
    """
    set_seed(SEED)
    device = torch.device(DEVICE)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    output_path = os.path.join(SUBMISSION_DIR, output_filename)

    # 1. Load Metadata to get Sample IDs
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_METADATA_PATH}")

    test_df = pd.read_csv(TEST_METADATA_PATH)
    sample_ids = test_df["sample_id"].tolist()

    # 2. Initialize Dataset and Loader
    print("Initializing test dataset...")
    test_dataset = GestureDataset(
        split="test", load_cached_data=load_cached_data, augment=False
    )

    # Verify alignment
    if len(test_dataset) != len(sample_ids):
        print(
            f"Warning: Dataset size ({len(test_dataset)}) does not match metadata size ({len(sample_ids)})."
        )
        # In a real scenario, we would need to handle this carefully.
        # Based on the provided metadata script, we expect exact matching.
        # We will proceed assuming the dataset contains the samples in the same order as metadata
        # (which is guaranteed by the sequential processing in data_loader).
        sample_ids = sample_ids[: len(test_dataset)]

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Crucial to keep order aligned with sample_ids
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 3. Run Prediction
    predictor = Predictor(checkpoint_path, device)
    print("Running inference...")
    predicted_sequences = predictor.predict(test_loader)

    # 4. Generate Submission
    print(f"Generating submission file at {output_path}...")

    with open(output_path, "w") as f:
        # No header required by the prompt example: "Session00001,2,12,3"
        # But standard CSVs often have headers. The prompt example does NOT show a header line.
        # "For instance: Session00001,2,12,3"
        # However, checking the sample submission or randomPredictions.csv in input might reveal a header.
        # input/randomPredictions.csv has header: "Id,Sequence" (but Id is int).
        # The prompt says: "Session00001,2,12,3". This looks like "SequenceID, LabelList".
        # I will follow the prompt's explicit format example strictly.

        for sid, seq in zip(sample_ids, predicted_sequences):
            # Join labels with commas
            labels_str = ",".join(map(str, seq))
            line = f"{sid},{labels_str}\n"
            f.write(line)

    print("Inference complete.")
