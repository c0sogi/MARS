import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_logger, seed_everything
from library.data_processing import DataHandler
from library.model import PAWDS, collate_fn

logger = get_logger("predict")


def generate_submission(
    model_path: str = Config.MODEL_SAVE_PATH,
    output_path: str = Config.SUBMISSION_PATH,
    batch_size: int = Config.BATCH_SIZE,
    debug_size: int = None,
    device: str = None,
):
    """
    Generates a submission file using a trained PA-WDS model.

    Args:
        model_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        debug_size (int): If set, limits the dataset size for debugging.
        device (str): Computation device ('cpu' or 'cuda').
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logger.info(f"Using device: {device}")

    # 2. Data Loading & Preprocessing
    # We must load training data to fit the scalers correctly,
    # ensuring the test data is transformed into the same feature space.
    if debug_size is not None:
        Config.DEBUG_SAMPLE_SIZE = debug_size
        logger.info(f"Debug mode enabled: limiting samples to {debug_size}")

    data_handler = DataHandler()
    # get_datasets handles caching and scaler fitting internally
    # We discard train and val datasets as we only need test for inference,
    # but we need the call to ensure scalers are fitted on train data.
    _, _, test_dataset = data_handler.get_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = PAWDS().to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    logger.info(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    predictions = []
    ids = []

    logger.info("Starting inference...")
    with torch.no_grad():
        for batch in test_loader:
            atomic_x = batch["atomic_features"].to(device)
            global_x = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            batch_ids = batch["ids"]

            # Forward pass
            outputs = model(atomic_x, global_x, mask)

            # Inverse Transformation
            # The model predicts log(1 + y), so we apply exp(y) - 1 to recover original scale.
            # We use torch.expm1 for numerical stability.
            preds_original_scale = torch.expm1(outputs).cpu().numpy()

            predictions.append(preds_original_scale)
            ids.extend(batch_ids)

    # Concatenate all batches
    if not predictions:
        logger.warning("No predictions generated. Check dataset.")
        return

    predictions = np.vstack(predictions)

    # 5. Submission Generation
    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Sort by ID to ensure consistent order
    submission_df = submission_df.sort_values("id")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")

    # Print sample for verification
    logger.info("Sample predictions:")
    print(submission_df.head())
