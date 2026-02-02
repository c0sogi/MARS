import os
import torch
from torch.utils.data import DataLoader
from library.config import Config, set_seed
from library.dataset import load_data, RNADataset
from library.model import HybridResNetBiGRU
from library.train import generate_submission


def run_prediction(
    model_path=Config.MODEL_SAVE_PATH,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    debug_size=Config.DEBUG_SUBSET_SIZE,
):
    """
    Main function to run the inference pipeline.

    Args:
        model_path (str): Path to the saved model weights.
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        debug (bool): If True, runs on a small subset of data.
        debug_size (int): Number of samples to use in debug mode.
    """
    # 1. Setup Environment
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running inference on device: {device}")

    # 2. Load Test Data
    # We manually load only the test data to avoid the overhead of loading train/val data
    # which happens if we use library.dataset.get_dataloaders
    print("Loading test data...")
    test_data = load_data(
        path=Config.TEST_DATA_PATH,
        cache_file="test_data.npz",
        load_cached_data=True,
        mode="test",
        debug=debug,
        debug_size=debug_size,
    )

    test_dataset = RNADataset(test_data, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Test dataset size: {len(test_dataset)}")

    # 3. Initialize Model
    print("Initializing model...")
    model = HybridResNetBiGRU()
    model.to(device)

    # 4. Load Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 5. Generate Submission
    # We use the existing generate_submission function from library.train
    # to ensure consistent formatting and logic.
    print("Generating submission...")
    generate_submission(model, test_loader, device, output_path)
    print("Inference completed successfully.")
