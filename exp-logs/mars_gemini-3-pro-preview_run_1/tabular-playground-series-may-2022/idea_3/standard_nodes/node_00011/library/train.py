import os
from library.config import Config
from library.utils import seed_everything
from library.model import train_model


def run_training(debug: bool = False, epochs: int = None):
    """
    Manages the training configuration and execution loop.

    Args:
        debug (bool): If True, uses a smaller subset of data for debugging purposes.
        epochs (int, optional): The number of training epochs. If provided, overrides
                                the default in Config.EPOCHS.
    """
    # 1. Update Configuration
    # Allow dynamic overriding of epochs for flexibility
    if epochs is not None:
        Config.EPOCHS = epochs

    # 2. Environment Setup
    # Initialize directories (working, submission) and set basic torch seeds
    Config.setup()

    # Apply comprehensive seeding (Python, NumPy, Torch, OS) for full reproducibility
    seed_everything(Config.SEED)

    print(f"Initializing training workflow (Debug={debug}, Epochs={Config.EPOCHS})...")

    # 3. Execute Training Pipeline
    # The train_model function from library.model encapsulates:
    # - Data loading and caching (via library.dataset)
    # - Model initialization (HybridTransformerModel)
    # - Training loop with AdamW, OneCycleLR, and BCEWithLogitsLoss
    # - Validation metrics tracking (AUC) and Early Stopping
    # - Final prediction on test set and submission file generation
    model = train_model(debug=debug)

    return model
