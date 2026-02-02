import torch
import os
from library.config import Config
from library.tokenizer import Tokenizer
from library.model import GFCN
from library.dataset import get_dataloaders
from library.utils import load_checkpoint
from library.train import generate_submission


def predict_test_set(load_cached_data=True, debug=False):
    """
    Loads the trained model and generates predictions for the test set.

    Args:
        load_cached_data (bool): Whether to load cached data artifacts (tokenizer, metadata).
        debug (bool): If True, enables debug mode (smaller dataset).
    """
    # Handle debug mode override
    if debug:
        print("Enabling DEBUG mode for inference...")
        Config.DEBUG = True

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Tokenizer
    # This handles loading the vocabulary from cache if available and requested
    tokenizer = Tokenizer(load_cached_data=load_cached_data)

    # Get Test DataLoader
    # get_dataloaders returns (train, val, test) tuple, we only need the test loader
    print("Initializing DataLoaders...")
    _, _, test_loader = get_dataloaders(tokenizer, load_cached_data=load_cached_data)

    # Initialize Model
    # The architecture must match the training configuration
    print("Initializing Model...")
    model = GFCN(num_classes=len(tokenizer)).to(device)

    # Load the best trained model weights
    # Config.BEST_MODEL_PATH points to the saved best model artifact
    print(f"Loading checkpoint from {Config.BEST_MODEL_PATH}...")
    load_checkpoint(model, filename=Config.BEST_MODEL_PATH)

    # Generate Submission
    # This function iterates through the loader, decodes predictions using the tokenizer,
    # and saves the results to Config.SUBMISSION_PATH
    generate_submission(model, test_loader, tokenizer, device)
