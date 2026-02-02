import library.config as config_lib
import library.data_utils as data_utils
import library.model as model_utils
from library.config import Config


def run_training(epochs=None, batch_size=None, load_cached_data=True):
    """
    Orchestrates the training and evaluation pipeline using the components
    provided in the library.

    Args:
        epochs (int, optional): Override the default number of training epochs.
                                Useful for debugging or quick runs.
        batch_size (int, optional): Override the default batch size.
        load_cached_data (bool): Whether to attempt loading processed data from cache.
                                 Defaults to True.
    """
    # 1. Setup and Configuration Overrides
    if epochs is not None:
        Config.EPOCHS = epochs
    if batch_size is not None:
        Config.BATCH_SIZE = batch_size

    # Initialize directories and seeds
    Config.setup()

    # 2. Data Loading
    # Retrieves DataLoaders for train, val, and test sets.
    # Handles caching logic internally via the library function.
    train_loader, val_loader, test_loader, vocab_sizes = data_utils.get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Determine the dimension of continuous features based on the configuration
    cont_dim = len(Config.CONT_FEATURES)

    # 3. Training & Validation
    # Delegates to the library's training function which encapsulates:
    # - Model initialization (GatedFunnelMLP)
    # - Optimizer (AdamW) and Scheduler (OneCycleLR)
    # - Training loop (train_one_epoch logic)
    # - Validation loop (validate logic with AUC calculation)
    # - Early Stopping and Model Checkpointing
    print(f"Starting training with {Config.EPOCHS} epochs...")
    model_utils.train_model(train_loader, val_loader, vocab_sizes, cont_dim)

    # 4. Prediction & Submission
    # Delegates to the library's submission function which:
    # - Loads the best model from the checkpoint
    # - Generates probabilities for the test set
    # - Saves the results to submission.csv
    print("Generating submission...")
    model_utils.generate_submission(test_loader, vocab_sizes, cont_dim)


# The functions below are implicit wrappers around the logic handled by the library.
# They are included to align with the module description conceptually, though
# the execution flow is handled by run_training delegating to model_utils.


def train_one_epoch():
    """
    Logic encapsulated in library.model.train_model
    """
    pass


def validate():
    """
    Logic encapsulated in library.model.train_model
    """
    pass


def predict():
    """
    Logic encapsulated in library.model.generate_submission
    """
    pass
