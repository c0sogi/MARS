import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.model import SCRNet
from library.data_loader import GestureDataset, collate_fn
from library.utils import set_seed, batch_decode


def generate_submission(
    checkpoint_path=Config.BEST_MODEL_PATH,
    output_path=Config.SUBMISSION_PATH,
    debug=Config.DEBUG,
):
    """
    Generates the submission file for the test dataset using the trained SCRNet model.

    Args:
        checkpoint_path (str): Path to the saved model weights.
        output_path (str): Path where the submission CSV will be saved.
        debug (bool): If True, runs on a small subset for testing.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating submission...")
    print(f"Model: {checkpoint_path}")
    print(f"Output: {output_path}")

    # 2. Data Loading
    # Load test metadata to get Sample IDs in order
    test_metadata = pd.read_csv(Config.TEST_METADATA_PATH)

    # Initialize Dataset and DataLoader
    test_dataset = GestureDataset(
        metadata_path=Config.TEST_METADATA_PATH, is_train=False, load_cached_data=True
    )

    if debug:
        # Subset for debugging
        indices = list(range(min(len(test_dataset), 10)))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        test_metadata = test_metadata.iloc[indices].reset_index(drop=True)
        print("Debug mode: Reduced dataset size.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Important: Must keep order to match metadata
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = SCRNet().to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    all_predictions = []

    with torch.no_grad():
        for skeletons, audios, _, lengths in test_loader:
            skeletons = skeletons.to(device)
            audios = audios.to(device)
            lengths = lengths.to(device)

            # Forward pass
            logits = model(skeletons, audios, lengths)

            # Decode batch
            # batch_decode handles Median Filtering, RLE, and Background removal
            batch_preds = batch_decode(logits, lengths)
            all_predictions.extend(batch_preds)

    # 5. Format and Save Submission
    # We assume test_loader order matches test_metadata order (shuffle=False)
    submission_lines = []

    if len(all_predictions) != len(test_metadata):
        print(
            f"Warning: Number of predictions ({len(all_predictions)}) does not match metadata length ({len(test_metadata)})."
        )

    for idx, row in test_metadata.iterrows():
        sample_id = row["sample_id"]

        if idx < len(all_predictions):
            pred_seq = all_predictions[idx]
            # Convert list of ints to comma-separated string
            pred_str = ",".join(map(str, pred_seq))
        else:
            pred_str = ""

        # Format: SessionID,Label1,Label2,...
        if pred_str:
            line = f"{sample_id},{pred_str}"
        else:
            line = f"{sample_id}"

        submission_lines.append(line)

    # Write to file
    with open(output_path, "w") as f:
        for line in submission_lines:
            f.write(line + "\n")

    print(f"Submission saved to {output_path}")
