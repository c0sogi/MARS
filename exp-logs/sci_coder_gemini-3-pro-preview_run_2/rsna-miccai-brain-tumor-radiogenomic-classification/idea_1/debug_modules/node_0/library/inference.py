import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

import library.config as config
import library.utils as utils
import library.data as data
import library.model as model


def generate_submission(
    checkpoint_path=config.CHECKPOINT_PATH,
    output_path=config.SUBMISSION_FILE,
    batch_size=config.BATCH_SIZE,
    device=config.DEVICE,
):
    """
    Generates the submission file for the test set using the trained model.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on.
    """
    # 1. Reproducibility
    utils.seed_everything(config.SEED)

    print(f"Using device: {device}")

    # 2. Load Test Metadata
    print(f"Loading test metadata from {config.TEST_METADATA_PATH}...")
    if not os.path.exists(config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata file not found at {config.TEST_METADATA_PATH}"
        )

    df_test = pd.read_csv(config.TEST_METADATA_PATH)
    print(f"Found {len(df_test)} test samples.")

    # 3. Dataset and DataLoader
    # Use 'val' transforms which typically include resizing and normalization but no augmentation
    test_dataset = data.BraTSDataset(
        metadata=df_test,
        base_dir=config.INPUT_DIR,
        transform=data.get_transforms("val"),
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Initialize Model
    print("Initializing model architecture...")
    # We set pretrained=False because we are about to load specific trained weights.
    # This prevents unnecessary downloads or errors if internet is restricted.
    net = model.MGMTNet(pretrained=False)
    net = net.to(device)

    # 5. Load Weights
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Checkpoint not found at {checkpoint_path}. Predictions will be based on random initialization."
        )

    # 6. Inference Loop
    net.eval()
    all_ids = []
    all_probs = []

    print("Starting inference...")
    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = net(inputs)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Move to CPU and collect
            probs_np = probs.cpu().numpy().flatten()
            ids_np = ids.numpy()

            all_ids.extend(ids_np)
            all_probs.extend(probs_np)

    # 7. Generate Submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": all_ids, "MGMT_value": all_probs})

    # 8. Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print("First 5 rows of submission:")
    print(submission_df.head())
