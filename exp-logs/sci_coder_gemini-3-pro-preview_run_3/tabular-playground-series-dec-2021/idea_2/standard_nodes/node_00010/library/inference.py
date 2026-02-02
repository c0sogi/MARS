import os
import torch
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.model import ResNetMLP, predict_and_submit


def generate_submission(
    model_path: str,
    output_path: str = "./submission/submission.csv",
    batch_size: int = 1024,
    num_workers: int = 4,
    device: str = None,
    num_blocks: int = 3,
    hidden_dim: int = 256,
):
    """
    Generates predictions for the test set using a trained model checkpoint.

    Args:
        model_path (str): Path to the saved model state dictionary (.pth file).
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker processes for data loading.
        device (str, optional): Device to run inference on ('cuda' or 'cpu').
                                If None, automatically detects.
    """
    # 1. Setup Device and Seed
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(42)

    print(f"Running inference on device: {device}")

    # 2. Prepare Data
    # We only need the test loader. get_dataloaders handles caching and preprocessing.
    # We pass load_cached_data=True to leverage existing preprocessed files if available.
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=True,
    )

    # 3. Initialize Model
    # Architecture must match the training configuration defined in train.py
    # Input features: 12 numerical + 44 binary = 56
    # Classes: 6 (mapped from original 7 classes, excluding class 5)
    input_dim = 56
    num_classes = 6

    model = ResNetMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        dropout_rate=0.2,
    )

    # 4. Load Model Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print(f"Loading model weights from {model_path}...")
    # Load state dict; map_location ensures it loads to the correct device initially
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move model to the computation device
    model.to(device)

    # 5. Generate and Save Predictions
    # predict_and_submit handles the inference loop, inverse mapping, and CSV saving.
    predict_and_submit(
        model=model,
        test_loader=test_loader,
        output_path=output_path,
        device=device,
    )
