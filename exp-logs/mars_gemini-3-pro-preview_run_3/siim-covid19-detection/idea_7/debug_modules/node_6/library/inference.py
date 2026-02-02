import os
import torch
from torch.utils.data import DataLoader
from library.config import Config, seed_everything
from library.dataset import ChestXRayDataset, collate_fn
from library.model import CovidCascadeRCNN
from library.engine import inference as engine_inference


def predict(checkpoint_path=None, batch_size=Config.BATCH_SIZE, debug=False):
    """
    Runs the inference pipeline on the test dataset.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint file.
                                         Defaults to Config.CHECKPOINT_DIR/best_model.pth.
        batch_size (int): Batch size for the dataloader. Defaults to Config.BATCH_SIZE.
        debug (bool): If True, runs inference on a small subset of the test data.
    """
    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Initialize Model
    model = CovidCascadeRCNN()
    model.to(device)

    # 3. Load Weights
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Checkpoint not found at {checkpoint_path}. Using initialized weights (random)."
        )

    # 4. Prepare Data Loader
    # We instantiate the dataset directly to support the 'debug' flag,
    # instead of using get_test_dataloader which hardcodes debug=False.
    test_ds = ChestXRayDataset(split="test", debug=debug)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 5. Run Inference
    # The engine_inference function handles TTA, WBF, hierarchical logic, and formatting.
    submission_path = Config.SUBMISSION_PATH

    # Ensure output directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    print(f"Starting inference on {len(test_ds)} images...")
    engine_inference(model, test_loader, device, submission_path)

    print(f"Inference completed successfully. Submission saved to {submission_path}")
