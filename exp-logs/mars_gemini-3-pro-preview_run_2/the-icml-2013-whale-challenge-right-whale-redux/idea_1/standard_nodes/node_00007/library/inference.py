import os
import torch
from library.config import Config
from library.model import WhaleResNet
from library.trainer import train_model, generate_submission


def run_inference(load_cached_data=False, num_epochs=None, debug=False):
    """
    Manages the inference process. It handles model training (if necessary or requested),
    loads the best performing weights, and generates the submission file for the test set.

    Args:
        load_cached_data (bool): If True, attempts to load a pre-trained model from the working directory.
                                 If False or if loading fails, the model is trained from scratch.
        num_epochs (int, optional): Number of epochs to train. Defaults to Config.NUM_EPOCHS.
        debug (bool): If True, enables debug mode (uses a subset of data).
    """
    # 1. Configuration Setup
    if debug:
        Config.DEBUG = True
        print("DEBUG mode enabled: Using subset of data.")

    if num_epochs is None:
        num_epochs = Config.NUM_EPOCHS

    # Ensure working directory exists for caching/artifacts
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    model = None

    # 2. Caching Logic: Attempt to load model if requested
    if load_cached_data:
        if os.path.exists(model_path):
            print(f"Attempting to load cached model from {model_path}...")
            try:
                # Initialize architecture
                model = WhaleResNet(pretrained=Config.PRETRAINED)
                # Load weights
                model.load_state_dict(
                    torch.load(model_path, map_location=Config.DEVICE)
                )
                model.to(Config.DEVICE)
                print("Cached model loaded successfully.")
            except Exception as e:
                print(f"Failed to load cached model: {e}")
                model = None
        else:
            print("No cached model found.")

    # 3. Training Logic: If model is missing or cache not used, train from scratch
    if model is None:
        print("Initiating model training...")
        # train_model handles the training loop, early stopping, and saves 'best_model.pth'
        # It returns the model with the best weights loaded.
        model = train_model(num_epochs=num_epochs)

    # 4. Prediction Logic: Generate submission
    if model is not None:
        # generate_submission handles the test DataLoader iteration, probability calculation,
        # and CSV export as per the requirements.
        generate_submission(model)
    else:
        print(
            "Error: Model could not be trained or loaded. Submission generation aborted."
        )
