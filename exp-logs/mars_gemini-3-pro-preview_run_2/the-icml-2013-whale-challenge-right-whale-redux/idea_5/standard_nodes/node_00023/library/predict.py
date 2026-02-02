import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.dataset import get_dataloaders
from library.model import WhaleEfficientNet


def generate_predictions(load_cached_data=True):
    """
    Generates predictions for the test set using the trained model.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache (.npy files)
                                 or re-process from scratch. Defaults to True.
    """
    # 1. Setup Environment
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Load Test Data
    # get_dataloaders returns (train, val, test). We only need test.
    # The caching logic is handled internally by library.dataset.load_dataset_data
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)
    print(f"Test loader initialized with {len(test_loader)} batches.")

    # 3. Initialize Model Architecture
    # We use pretrained=False here because we are about to load our specific trained weights.
    # The architecture parameters (backbone, channels, etc.) are pulled from Config inside the class.
    print(f"Initializing model architecture: {Config.MODEL_NAME}")
    model = WhaleEfficientNet(pretrained=False)
    model = model.to(device)

    # 4. Load Trained Weights
    checkpoint_path = Config.MODEL_PATH
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = load_checkpoint(model, path=checkpoint_path, device=Config.DEVICE)

        if checkpoint:
            # Print full precision metrics as requested
            epoch = checkpoint.get("epoch", "Unknown")
            val_score = checkpoint.get("val_score", "Unknown")
            print(
                f"Model loaded successfully. Training Epoch: {epoch}, Validation AUC: {val_score}"
            )
    else:
        print(f"WARNING: Checkpoint file not found at {checkpoint_path}.")
        print(
            "Proceeding with random/initialized weights (Predictions will be meaningless)."
        )

    # 5. Run Inference
    model.eval()
    all_clips = []
    all_probs = []

    print("Starting inference on test set...")
    with torch.no_grad():
        for inputs, clips in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)

            # Apply sigmoid to get probabilities (0-1)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_clips.extend(clips)
            all_probs.extend(probs)

    # 6. Generate Submission File
    submission_df = pd.DataFrame({"clip": all_clips, "probability": all_probs})

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Predictions generated for {len(submission_df)} clips.")
    print(f"Submission file saved to: {Config.SUBMISSION_PATH}")

    return submission_df
