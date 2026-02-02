import os
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.data_loader import get_dataloaders
from library.model import CompositeCNN, generate_submission, set_seeds


def predict(load_cached_data=True, batch_size=32, max_samples=None):
    """
    Loads the best trained model and generates a submission file for the test set.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        batch_size (int): Batch size for inference.
        max_samples (int, optional): Limit the number of test samples for debugging.
    """
    # 1. Initialize Configuration
    config = Config()
    config.BATCH_SIZE = batch_size

    # Set seeds for reproducibility
    set_seeds(config.SEED)

    # 2. Load Data
    # We only require the test_loader. get_dataloaders handles caching internally.
    print(f"Loading test data (Cached: {load_cached_data})...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Adjust DataLoader if necessary
    # If the requested batch_size differs from the default in Config, or if max_samples is set,
    # we need to reconstruct the DataLoader.
    default_batch_size = Config().BATCH_SIZE

    if batch_size != default_batch_size or max_samples is not None:
        print(
            f"Reconfiguring test loader: Batch Size={batch_size}, Max Samples={max_samples}"
        )
        test_dataset = test_loader.dataset

        # Apply max_samples limit if requested
        if max_samples is not None:
            limit = min(len(test_dataset.images), max_samples)
            test_dataset.images = test_dataset.images[:limit]
            test_dataset.angles = test_dataset.angles[:limit]
            # Note: test_dataset.labels is None

        # Re-create the DataLoader with the specific batch size and subset
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    # 4. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing model on {device}...")
    model = CompositeCNN(config).to(device)

    # 5. Load Model Weights
    checkpoint_path = config.MODEL_CHECKPOINT
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Please ensure the training process has completed successfully."
        )

    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 6. Generate Submission
    # This function handles the inference loop and saving the CSV to config.SUBMISSION_FILE
    print("Generating submission...")
    generate_submission(model, test_loader, config)
    print("Prediction process completed.")
