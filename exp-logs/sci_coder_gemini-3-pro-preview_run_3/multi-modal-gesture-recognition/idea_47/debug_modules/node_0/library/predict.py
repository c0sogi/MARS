import os
import torch
import numpy as np
from library.config import Config
from library.utils import set_seeds, decode_predictions_to_sequence, save_submission
from library.data_loader import get_dataloaders
from library.model import DGC_KN


def generate_submission(
    model_path=Config.BEST_MODEL_PATH, output_path=Config.SUBMISSION_PATH
):
    """
    Loads the best trained model, runs inference on the test dataset,
    decodes the predictions, and generates the final submission CSV file.

    Args:
        model_path (str): Path to the saved model checkpoint.
        output_path (str): Path where the submission CSV will be saved.
    """
    # 1. Setup
    set_seeds()
    device = Config.DEVICE

    print(f"Preparing to generate submission using model at: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Please run training first."
        )

    # 2. Initialize Model
    # We instantiate the architecture and load the state dictionary
    model = DGC_KN()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Load Test Data
    # get_dataloaders returns (train, val, test). We only need test.
    # This will handle caching automatically via the library implementation.
    print("Loading test dataset...")
    _, _, test_loader = get_dataloaders(debug=False)

    predictions = {}

    print(f"Starting inference on {len(test_loader)} test sequences...")

    # 4. Inference Loop
    with torch.no_grad():
        for i, (features, _, sample_ids) in enumerate(test_loader):
            # features shape: (Batch=1, Time, InputDim)
            features = features.to(device)

            # Forward pass
            outputs = model(features)

            # Extract probabilities from the final stage (Stage 3)
            # The model output dictionary contains "probs_3" which is softmax(logits_3)
            # Shape: (Batch=1, Time, NumClasses)
            probs_batch = outputs["probs_3"]

            # Remove batch dimension to get (Time, NumClasses)
            probs = probs_batch.squeeze(0).cpu().numpy()

            # Decode frame-wise probabilities to gesture sequence
            # This applies: Argmax -> Run-Length Encoding -> Min Duration Filter -> Background Removal
            sequence = decode_predictions_to_sequence(probs)

            # Store result
            # sample_ids comes as a tuple from the dataloader (due to batching), take the first element
            sample_id = sample_ids[0]
            predictions[sample_id] = sequence

            # Optional: Simple progress indicator
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(test_loader)} samples")

    # 5. Save Submission
    print(f"Saving predictions to {output_path}...")
    save_submission(predictions, output_path)
    print("Submission generation complete.")
