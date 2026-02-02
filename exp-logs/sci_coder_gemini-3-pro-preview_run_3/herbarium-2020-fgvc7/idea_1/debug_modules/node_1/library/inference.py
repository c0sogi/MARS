import os
import torch
import library.config as config
import library.dataset as dataset
import library.model as model_lib


def run_inference(
    checkpoint_path=config.MODEL_SAVE_PATH,
    batch_size=config.BATCH_SIZE,
    device=config.DEVICE,
    num_workers=config.NUM_WORKERS,
):
    """
    Loads the test dataset and a trained model checkpoint to generate submission predictions.

    Args:
        checkpoint_path (str): Path to the saved model state dictionary.
        batch_size (int): Batch size for inference.
        device (str): Computation device ('cuda' or 'cpu').
        num_workers (int): Number of subprocesses for data loading.
    """
    # Set random seed for reproducibility
    config.set_seed(config.SEED)

    # Update configuration with runtime arguments to ensure library functions use them
    config.BATCH_SIZE = batch_size
    config.NUM_WORKERS = num_workers
    config.DEVICE = device

    print(f"Starting inference on device: {device}")

    # Load DataLoaders
    # We rely on the library function which handles caching of training weights.
    # We only need the test_loader for inference.
    print("Loading dataloaders...")
    _, _, test_loader = dataset.get_dataloaders(load_cached_data=True)

    # Initialize the model architecture
    # pretrained=False avoids downloading ImageNet weights since we load a checkpoint immediately
    print("Initializing model...")
    model = model_lib.ResNet18Classifier(
        num_classes=config.NUM_CLASSES, pretrained=False
    )

    # Load the trained weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move model to the specified device
    model = model.to(device)

    # Generate predictions and save submission file
    # The generate_submission function handles the loop and CSV writing based on config.SUBMISSION_PATH
    model_lib.generate_submission(model, test_loader)
