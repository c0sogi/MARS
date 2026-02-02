import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.dataset import load_metadata, get_transforms, PathologyDataset
from library.models import get_model
from library.engine import predict_with_tta


def predict_ensemble(
    output_path: str = Config.SUBMISSION_PATH,
    device: str = Config.DEVICE,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
):
    """
    Generates predictions for the test set using a heterogeneous ensemble of models.

    The function performs the following steps:
    1. Loads test metadata and prepares the data loader.
    2. Iterates through each model architecture defined in Config.MODELS.
    3. Loads the best trained weights for each model.
    4. Generates predictions using 4-view Test Time Augmentation (TTA).
    5. Aggregates predictions via soft voting (averaging).
    6. Saves the final predictions to a CSV file.

    Args:
        output_path (str): Path to save the submission CSV. Defaults to Config.SUBMISSION_PATH.
        device (str): Computation device ('cuda' or 'cpu'). Defaults to Config.DEVICE.
        batch_size (int): Batch size for inference. Defaults to Config.BATCH_SIZE.
        num_workers (int): Number of dataloader workers. Defaults to Config.NUM_WORKERS.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    print(f"Starting Ensemble Inference on device: {device}")

    # --- 1. Data Preparation ---
    # Load test metadata (uses caching mechanism internally)
    df_test = load_metadata(phase="test")

    # Prepare Dataset and DataLoader
    # We use the 'test' transforms which include normalization and center crop
    test_dataset = PathologyDataset(
        df=df_test, phase="test", transform=get_transforms(phase="test")
    )

    # Shuffle must be False to maintain alignment with df_test IDs
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f"Test set size: {len(df_test)} images")

    # --- 2. Ensemble Prediction ---
    # Initialize accumulator for soft voting
    ensemble_probs = np.zeros(len(df_test), dtype=np.float64)
    models_used = 0

    for model_name in Config.MODELS:
        print(f"Processing model: {model_name}")

        # Construct path to the saved weights
        # Assuming weights are saved as {model_name}_best.pth in the working directory
        weights_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights file not found for {model_name} at {weights_path}"
            )

        # Instantiate model architecture
        model = get_model(model_name, pretrained=False)

        # Load weights
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            # Handle case where checkpoint might be a dict containing 'model_state_dict'
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to load weights for {model_name}: {e}")

        model.to(device)
        model.eval()

        # Generate predictions with TTA
        # predict_with_tta returns a numpy array of probabilities (N, 1) or (N,)
        preds = predict_with_tta(model, test_loader, device)

        # Ensure preds is 1D array
        preds = preds.flatten()

        # Accumulate
        ensemble_probs += preds
        models_used += 1

        # Clean up to save memory
        del model
        torch.cuda.empty_cache()

    if models_used == 0:
        raise RuntimeError("No models were successfully processed for the ensemble.")

    # --- 3. Aggregation ---
    # Compute average probability (Soft Voting)
    avg_probs = ensemble_probs / models_used

    # --- 4. Submission Generation ---
    submission_df = pd.DataFrame({"id": df_test["id"], "label": avg_probs})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print("Inference complete.")
