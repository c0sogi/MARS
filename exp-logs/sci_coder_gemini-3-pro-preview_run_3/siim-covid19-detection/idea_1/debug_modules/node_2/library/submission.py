import os
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import SIIMDataset
from library.model import MultiTaskFasterRCNN
from library.utils import collate_fn
from library.engine import predict, set_seed


def generate_submission(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, num_samples=None
):
    """
    Generates the submission file for the competition.

    Args:
        load_cached_data (bool): Whether to load dataset from cache.
        batch_size (int): Batch size for inference.
        num_samples (int, optional): Number of samples to use (for debugging).
    """
    # 1. Set Seed for Reproducibility
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 2. Load Test Data
    print("Initializing Test Dataset...")
    test_dataset = SIIMDataset(split="test", load_cached_data=load_cached_data)

    # Optional: Subset for debugging
    if num_samples is not None:
        print(f"Subsetting test dataset to {num_samples} samples.")
        test_dataset.df = test_dataset.df.iloc[:num_samples].reset_index(drop=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
    )

    # 3. Load Model
    print("Initializing Model...")
    model = MultiTaskFasterRCNN()

    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading model weights from {Config.MODEL_SAVE_PATH}...")
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Model checkpoint not found at {Config.MODEL_SAVE_PATH}. Using random weights."
        )

    model.to(device)

    # 4. Run Inference and Generate Submission
    # The predict function in library.engine handles the loop, formatting, and saving.
    predict(
        model=model,
        test_loader=test_loader,
        test_df=test_dataset.df,
        device=device,
        submission_path=Config.SUBMISSION_PATH,
    )
