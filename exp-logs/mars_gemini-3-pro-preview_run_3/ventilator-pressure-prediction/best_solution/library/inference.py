import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import prepare_datasets, VentilatorDataset
from library.model import CWDHNet


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility during inference.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_predictions(batch_size=Config.BATCH_SIZE):
    """
    Generates predictions for the test set using the best trained model
    and saves the result to the submission file defined in Config.
    """
    set_seed()
    device = torch.device(Config.DEVICE)
    print(f"Inference using device: {device}")

    # 1. Load Data
    # We invoke prepare_datasets to ensure test_x is available and scaled correctly.
    # We rely on the caching mechanism to avoid re-processing if training just finished.
    print("Loading test data...")
    _, _, _, _, test_x = prepare_datasets(load_cached_data=True)

    # Load test_ids from cache (created by prepare_datasets)
    # These correspond to the flattened ID structure required for submission
    test_ids_path = os.path.join(Config.WORKING_DIR, "test_ids.npy")
    if not os.path.exists(test_ids_path):
        raise FileNotFoundError(
            f"test_ids.npy not found at {test_ids_path}. "
            "Please ensure data preparation has been run."
        )
    test_ids = np.load(test_ids_path)

    # Create Dataset and Loader
    test_dataset = VentilatorDataset(test_x, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 2. Load Model
    print("Loading model...")
    model = CWDHNet().to(device)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    # Load weights
    checkpoint = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 3. Inference Loop
    print("Generating predictions...")
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            preds = model(inputs)

            # Move to CPU and flatten immediately to save GPU memory
            # preds shape: (batch_size, 80) -> flatten -> (batch_size * 80,)
            predictions.append(preds.cpu().numpy().flatten())

    # Concatenate all batches
    predictions = np.concatenate(predictions)

    # Flatten IDs to match predictions (N_breaths, 80) -> (N_breaths * 80,)
    flat_ids = test_ids.flatten()

    # 4. Validation and saving
    # Ensure lengths match
    if len(flat_ids) != len(predictions):
        print(
            f"Warning: Length mismatch! IDs: {len(flat_ids)}, Preds: {len(predictions)}"
        )
        # Truncate to the minimum length to allow saving, though this indicates an upstream error
        min_len = min(len(flat_ids), len(predictions))
        flat_ids = flat_ids[:min_len]
        predictions = predictions[:min_len]

    # Construct DataFrame
    submission_df = pd.DataFrame({"id": flat_ids, "pressure": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_FILE}")
    print(f"Total predictions: {len(submission_df)}")
