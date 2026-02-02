import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import DP_GI_BiLSTM


def predict(load_cached_data=True):
    """
    Loads the trained model, performs inference on the test set,
    and generates the submission file.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
                                 Passed to get_dataloaders.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Inference device: {device}")

    # 2. Load Data
    # get_dataloaders handles the caching logic internally.
    # We only need the test_loader for inference.
    print(f"Loading test data (load_cached_data={load_cached_data})...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Load Model
    print("Loading model...")
    model = DP_GI_BiLSTM(input_dim=Config.INPUT_DIM).to(device)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Please train the model first."
        )

    # Load weights
    checkpoint = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 4. Inference Loop
    print("Starting inference...")
    predictions = []

    with torch.no_grad():
        for inputs, _, _ in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            # Output shape: (Batch_Size, Seq_Len)
            preds = model(inputs)

            # Flatten to (Batch_Size * Seq_Len) and move to CPU
            preds_flat = preds.view(-1).cpu().numpy()
            predictions.extend(preds_flat)

    predictions = np.array(predictions)
    print(f"Total predictions generated: {len(predictions)}")

    # 5. Generate Submission
    print("Processing submission...")

    # Load test metadata to map predictions to IDs
    if not os.path.exists(Config.TEST_META):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_META}")

    test_meta = pd.read_csv(Config.TEST_META)

    # CRITICAL ALIGNMENT STEP:
    # The VentilatorDataset (via library.dataset.add_features) sorts the input dataframe
    # by ['breath_id', 'id'] to ensure correct sequence formation.
    # Therefore, the predictions array corresponds to rows in that specific order.
    # We must sort the metadata similarly to map predictions to the correct IDs.
    test_meta_sorted = test_meta.sort_values(by=["breath_id", "id"])

    # Check for consistency
    if len(predictions) != len(test_meta_sorted):
        print(
            f"Warning: Prediction count ({len(predictions)}) does not match Metadata count ({len(test_meta_sorted)})."
        )

    # Assign predictions
    test_meta_sorted["pressure"] = predictions

    # Sort back by 'id' for the final submission format
    submission = test_meta_sorted.sort_values(by="id")[["id", "pressure"]]

    # 6. Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
