import torch
import os
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import GNSSWindowDataset
from library.model import BiLSTMRegressor, generate_submission
from library.train import set_seed


def run_inference(batch_size=Config.BATCH_SIZE, load_cached_data=True, model_path=None):
    """
    Orchestrates the inference pipeline: loads the test dataset, loads the trained model,
    and generates the submission file using the library functions.

    Args:
        batch_size (int): Batch size for inference.
        load_cached_data (bool): If True, attempts to load preprocessed data from cache.
                                 If False or cache missing, re-computes data.
        model_path (str): Path to the trained model weights. If None, uses default cache path.
    """
    # 1. Setup Environment
    set_seed(Config.RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}...")

    # 2. Data Loading
    # The GNSSWindowDataset internally handles caching via preprocessing.load_dataset
    print("Initializing Test Dataset...")
    test_dataset = GNSSWindowDataset(mode="test", load_cached_data=load_cached_data)

    # Shuffle must be False for test data to align with submission IDs
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Bi-LSTM Model...")
    # Determine input dimensions dynamically from the dataset features
    input_dim = test_dataset.features.shape[1]
    output_dim = len(Config.TARGET_COLUMNS)

    model = BiLSTMRegressor(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        output_dim=output_dim,
    )

    # 4. Load Model Weights
    if model_path is None:
        model_path = os.path.join(Config.CACHE_DIR, "bilstm_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Please train the model first."
        )

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5. Generate Submission
    # generate_submission handles the prediction loop, coordinate reconstruction, and file saving
    print("Generating Submission...")
    generate_submission(model, test_loader)

    print("Inference pipeline completed successfully.")
