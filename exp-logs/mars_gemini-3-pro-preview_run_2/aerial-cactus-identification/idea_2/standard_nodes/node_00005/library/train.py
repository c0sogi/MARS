import torch
import os
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import CactusResNet, train_model, predict_and_submit
from library.utils import load_checkpoint


def run_training(num_epochs=Config.NUM_EPOCHS, debug=Config.DEBUG):
    """
    Executes the training pipeline.

    Args:
        num_epochs (int): Number of training epochs.
        debug (bool): If True, uses a small subset of data for debugging.
    """
    # 1. Setup System
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing training on {device}...")
    print(f"Debug Mode: {debug}")

    # 2. Prepare Data
    # get_dataloaders handles loading metadata and caching images via _load_and_cache_data
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # 3. Initialize Model
    model = CactusResNet()

    # 4. Train Model
    # train_model handles the training loop, validation, early stopping, and saving the best checkpoint.
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )


def run_inference(debug=Config.DEBUG):
    """
    Executes the inference pipeline using the best saved model.

    Args:
        debug (bool): If True, uses a small subset of data for debugging.
    """
    # 1. Setup System
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing inference on {device}...")

    # 2. Prepare Data
    # We only need the test loader for inference
    _, _, test_loader = get_dataloaders(debug=debug)

    # 3. Initialize Model
    model = CactusResNet()

    # 4. Load Best Weights
    checkpoint_path = Config.OUTPUT_MODEL_PATH
    if not os.path.exists(checkpoint_path):
        print(
            f"Error: Checkpoint not found at {checkpoint_path}. Please run training first."
        )
        return

    print(f"Loading best model from {checkpoint_path}...")
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    print(f"Model loaded. Best Validation AUC: {checkpoint.get('best_auc', 'N/A')}")

    # 5. Predict and Submit
    # predict_and_submit handles Test Time Augmentation (TTA) and CSV generation
    predict_and_submit(model, test_loader, device)


def main():
    """
    Main entry point for the training and inference process.
    """
    # Execute Training
    run_training()

    # Execute Inference
    run_inference()
