import torch
import os
from library.config import (
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE,
    DEBUG,
    DEBUG_SIZE,
    DROPOUT_RATE,
)
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import SAHCN, predict_and_submit


def generate_predictions(
    model_path=MODEL_SAVE_PATH,
    submission_path=SUBMISSION_PATH,
    batch_size=BATCH_SIZE,
    debug=DEBUG,
    debug_size=DEBUG_SIZE,
    load_cached_data=True,
):
    """
    Loads a trained model and generates predictions for the test set.

    Args:
        model_path (str): Path to the saved model state dictionary.
        submission_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        debug (bool): If True, runs on a small subset of the test data.
        debug_size (int): Number of samples to use in debug mode.
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # 1. Set Random Seeds for reproducibility
    seed_everything()

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Load Data
    # We only need the test_loader for prediction
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        load_cached_data=load_cached_data,
        batch_size=batch_size,
        debug=debug,
        debug_size=debug_size,
    )

    # 4. Initialize Model Architecture
    # Dropout rate is required for initialization but unused during inference (eval mode)
    model = SAHCN(dropout_rate=DROPOUT_RATE)

    # 5. Load Trained Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    print(f"Loading model weights from {model_path}...")
    # map_location ensures weights are loaded to the correct device (e.g. CPU if no GPU)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 6. Run Inference and Generate Submission
    # predict_and_submit handles setting eval mode, the inference loop, and saving to CSV
    predict_and_submit(
        model=model,
        test_loader=test_loader,
        device=device,
        submission_path=submission_path,
    )
