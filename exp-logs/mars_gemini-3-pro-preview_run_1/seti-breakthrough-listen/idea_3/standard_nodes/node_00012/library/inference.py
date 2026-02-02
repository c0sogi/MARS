import os
import sys
import pandas as pd
import torch
import numpy as np

from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.model import TechnosignatureModel
from library.data import get_dataloaders


def predict(debug=False):
    """
        Performs inference on the test set using the trained TechnosignatureModel
        and generates a submission file.
    >>>>>>> REPLACE
    <<<<<<< SEARCH
        # 3. Model Initialization
        # We initialize the model architecture. Pretrained is set to False as we will load our own weights.
        print("Initializing model...")
        model = SpatiotemporalResNet(pretrained=False)
        model = model.to(device)
    =======
        # 3. Model Initialization
        # We initialize the model architecture. Pretrained is set to False as we will load our own weights.
        print("Initializing model...")
        model = TechnosignatureModel(pretrained=False)
        model = model.to(device)

        Args:
            debug (bool): If True, runs in debug mode. Note that based on the data loader
                          implementation, the test set size is typically preserved even in
                          debug mode to ensure a valid submission format.
    """
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting inference on device: {device}")

    # 2. Data Loading
    # We utilize the centralized data loader function to ensure consistent preprocessing.
    # We are only interested in the test_loader (the third return value).
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=debug
    )

    # 3. Model Initialization
    # We initialize the model architecture. Pretrained is set to False as we will load our own weights.
    print("Initializing model...")
    model = SpatiotemporalResNet(pretrained=False)
    model = model.to(device)

    # 4. Load Checkpoint
    # We load the best model saved during training.
    checkpoint_path = Config.MODEL_SAVE_PATH
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. Cannot proceed with inference."
        )

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = load_checkpoint(model, filename=checkpoint_path, device=device)

    # Log validation score if available in checkpoint
    if checkpoint and "score" in checkpoint:
        print(f"Checkpoint loaded. Best Validation AUC: {checkpoint['score']}")

    # 5. Inference Loop
    model.eval()
    ids = []
    predictions = []

    print("Running prediction loop...")
    with torch.no_grad():
        for batch_inputs, batch_ids in test_loader:
            # Move inputs to the configured device
            batch_inputs = batch_inputs.to(device)

            # Forward pass
            logits = model(batch_inputs)

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Move to CPU and flatten to 1D array
            probs_np = probs.cpu().numpy().flatten()

            # Store results
            ids.extend(batch_ids)
            predictions.extend(probs_np)

    # 6. Submission Generation
    print(f"Generating submission file with {len(ids)} predictions...")
    submission_df = pd.DataFrame({"id": ids, "target": predictions})

    # Ensure the submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
