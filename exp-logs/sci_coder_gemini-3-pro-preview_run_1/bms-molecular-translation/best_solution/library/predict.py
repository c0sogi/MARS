import os
import torch
from library.config import Config
from library.data import get_dataloaders
from library.model import AttributeConditionedModel
from library.train import generate_submission


def generate_predictions(debug=False):
    """
    Loads the trained model and performs inference on the test dataset.
    Generates InChI strings via greedy decoding and saves the submission file.

    Args:
        debug (bool): If True, limits the dataset size for faster debugging.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Data
    # get_dataloaders handles metadata loading, processing, caching, and tokenizer creation.
    # We only require the test_loader and the tokenizer for decoding.
    # The load_cached_data=True flag ensures efficient loading from parquet caches.
    debug_size = 100 if debug else None
    _, _, test_loader, tokenizer = get_dataloaders(
        load_cached_data=True, debug_size=debug_size
    )

    # 3. Initialize Model
    # The AttributeConditionedModel includes the MobileNet encoder, attribute head, and GRU decoder.
    model = AttributeConditionedModel().to(device)

    # 4. Load Pre-trained Weights
    # We load the state dictionary saved during the training process.
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model weights from: {Config.BEST_MODEL_PATH}")
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.BEST_MODEL_PATH}. Using random weights."
        )

    # 5. Generate and Save Predictions
    # generate_submission handles the inference loop, decoding, and CSV export.
    generate_submission(test_loader, model, tokenizer, device)
