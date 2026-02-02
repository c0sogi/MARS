import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

from library.config import Config
from library.model import CustomModel
from library.data import prepare_loaders
from library.utils import set_seed


def predict(load_cached_data=True):
    """
    Loads the trained model, performs inference on the test set, and saves the submission file.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
                                 If False, re-processes data from metadata CSVs.
    """
    # 1. Setup
    config = Config()
    set_seed(config.seed)
    device = config.device
    print(f"Inference running on: {device}")

    # 2. Prepare Data
    print("Loading tokenizer and preparing data loaders...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # We force debug=False to ensure we predict on the full test set
    _, _, test_loader = prepare_loaders(
        tokenizer=tokenizer, load_cached_data=load_cached_data, debug=False
    )

    # 3. Load Model
    print("Initializing model architecture...")
    model = CustomModel()

    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {config.model_path}")

    print(f"Loading model weights from {config.model_path}...")
    state_dict = torch.load(config.model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 4. Inference Loop
    print("Starting inference...")
    all_preds = []

    with torch.no_grad():
        for data in test_loader:
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)

            # Forward pass
            outputs = model(input_ids, attention_mask)

            # Flatten outputs: (batch_size, 1) -> (batch_size,)
            outputs = outputs.view(-1)

            all_preds.append(outputs.cpu().numpy())

    # 5. Post-Processing
    if len(all_preds) == 0:
        predictions = np.array([])
    else:
        predictions = np.concatenate(all_preds)

    # Clip predictions to valid range [0, 1]
    predictions = np.clip(predictions, 0, 1)

    # 6. Generate Submission
    print("Generating submission file...")

    # Load test metadata to get the IDs
    if not os.path.exists(config.test_metadata_path):
        raise FileNotFoundError(
            f"Test metadata not found at {config.test_metadata_path}"
        )

    df_test = pd.read_csv(config.test_metadata_path)

    # Sanity check
    if len(df_test) != len(predictions):
        raise ValueError(
            f"Size mismatch: Test set has {len(df_test)} rows but generated {len(predictions)} predictions."
        )

    # Create submission DataFrame
    submission_df = pd.DataFrame({"id": df_test["id"], "score": predictions})

    # Save to disk
    # Config.submission_dir is created in Config.__init__, but we ensure specific path safety
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)
    submission_df.to_csv(config.submission_path, index=False)

    print(f"Submission saved successfully to {config.submission_path}")
    print("First 5 predictions:")
    print(submission_df.head())
