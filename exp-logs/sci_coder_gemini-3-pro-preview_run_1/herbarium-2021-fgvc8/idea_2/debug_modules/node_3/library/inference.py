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

    # Default to inferred classes
    num_classes = len(idx2cat)
    state_dict = None

    # 2. Load Weights & Config (Cite debug_lesson_2)
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict) and "classes" in checkpoint:
            # Use configuration from checkpoint
            classes = checkpoint["classes"]
            idx2cat = {i: cat for i, cat in enumerate(classes)}
            num_classes = len(classes)
            state_dict = checkpoint["state_dict"]
            print(f"Loaded configuration from checkpoint: {num_classes} classes")
        else:
            # Fallback for legacy checkpoints
            state_dict = checkpoint
            print(
                f"Warning: No config in checkpoint. Using inferred {num_classes} classes."
            )
    else:
        print(
            f"Warning: Checkpoint file {checkpoint_path} not found! Predictions will be random."
        )

    # 3. Initialize Model
    model = HerbariumEfficientNet(num_classes=num_classes)

    if state_dict is not None:
        model.load_state_dict(state_dict)

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
