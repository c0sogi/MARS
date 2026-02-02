import torch
from library.config import Config
from library.dataset import get_data_loaders
from library.model import WSDHNet, predict
from library.utils import get_device


def generate_predictions(load_cached_data=True):
    """
    Manages the inference pipeline on the test dataset.

    This function:
    1. Initializes the environment and configuration.
    2. Loads the test data (using caching if requested).
    3. Dynamically determines the input dimension from the data.
    4. Initializes the WSDH-Net model architecture.
    5. Delegates to library.model.predict to load weights, run inference, and save the submission.

    Args:
        load_cached_data (bool): Whether to attempt loading pre-processed data from cache.
                                 If False, data is re-processed from raw CSVs.
    """
    # 1. Initialize Configuration (creates directories if needed)
    Config.initialize()

    # 2. Setup Device (GPU/CPU)
    device = get_device()

    # 3. Load Data
    # get_data_loaders returns (train_loader, val_loader, test_loader).
    # We only need the test_loader for inference.
    # The caching logic is handled internally by library.dataset.prepare_data
    _, _, test_loader = get_data_loaders(load_cached_data=load_cached_data)

    # 4. Determine Input Dimension
    # Fetch a single batch to inspect feature size: [Batch, Seq_Len, Features]
    # This ensures the model input layer matches the processed data features.
    sample_x, _, _ = next(iter(test_loader))
    input_dim = sample_x.shape[2]

    # 5. Initialize Model Architecture
    # We instantiate the model structure here. The weights will be loaded inside the predict function.
    model = WSDHNet(input_dim=input_dim).to(device)

    # 6. Generate Predictions
    # The predict function from library.model handles:
    # - Loading the state_dict from Config.MODEL_PATH
    # - Batched inference with torch.no_grad()
    # - Flattening predictions
    # - Formatting and saving the CSV to Config.SUBMISSION_PATH
    predict(model, test_loader, device)
