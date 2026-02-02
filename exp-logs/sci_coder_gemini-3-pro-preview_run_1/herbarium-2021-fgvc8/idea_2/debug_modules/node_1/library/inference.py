import os
import torch
from library.utils import seed_everything
from library.dataset import get_test_dataloader
from library.model import HerbariumEfficientNet
from library.trainer import Trainer


def predict_and_submit(
    checkpoint_path="./working/idea_2/model.pth",
    output_file="./submission/submission.csv",
    batch_size=32,
    num_workers=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    debug=False,
):
    """
    Loads a trained model, generates predictions for the test set, and saves the submission file.

    Args:
        checkpoint_path (str): Path to the trained model weights.
        output_file (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker threads for data loading.
        device (str): Device to run inference on ('cpu' or 'cuda').
        debug (bool): If True, runs on a small subset of the test data.
    """
    # Ensure reproducibility
    seed_everything(42)

    print(f"Starting inference on device: {device}")

    # 1. Prepare Data
    # get_test_dataloader returns the loader and the index-to-category mapping needed for decoding
    test_loader, idx2cat = get_test_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=True,
        debug=debug,
    )

    num_classes = len(idx2cat)
    print(f"Number of classes detected: {num_classes}")

    # 2. Initialize Model
    model = HerbariumEfficientNet(num_classes=num_classes)

    # 3. Load Weights
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint file {checkpoint_path} not found! Predictions will be random."
        )

    # 4. Run Prediction
    # We use the Trainer class which already implements the prediction loop and CSV saving.
    # We pass None for criterion, optimizer, and scheduler as they are not needed for inference.
    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=None,
        scheduler=None,
        device=device,
        save_dir=os.path.dirname(checkpoint_path),
    )

    trainer.predict(test_loader, idx2cat, output_file=output_file)
