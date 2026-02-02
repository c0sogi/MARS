import os
import torch
from torch.utils.data import DataLoader
from library.config import (
    TEST_METADATA_PATH,
    CACHE_TEST_X,
    CACHE_TEST_IDS,
    MODEL_SAVE_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import seed_everything, get_device
from library.data import generate_dataset, BraTSDataset
from library.model import SliceGroupedFusionNet
from library.train import generate_submission as run_inference_and_save


def generate_submission(load_cached_data=True):
    """
    High-level function to orchestrate the submission generation process.
    Loads the trained model and test dataloader, then runs inference.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    # 1. Setup
    seed_everything(SEED)
    device = get_device()

    # 2. Data Loading
    # We use generate_dataset directly to load only the test set
    X_test, _, ids_test = generate_dataset(
        metadata_path=TEST_METADATA_PATH,
        cache_x_path=CACHE_TEST_X,
        cache_ids_path=CACHE_TEST_IDS,
        load_cached_data=load_cached_data,
    )

    test_dataset = BraTSDataset(X_test, ids=ids_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = SliceGroupedFusionNet()
    model.to(device)

    # 4. Load Trained Weights
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print(
            f"Warning: Model weights not found at {MODEL_SAVE_PATH}. Using initialized weights."
        )

    # 5. Run Inference and Save
    # Ensure the output directory exists as per requirements
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    output_path = os.path.join(submission_dir, "submission.csv")

    # Use the logic from library.train to generate predictions and save to CSV
    run_inference_and_save(model, test_loader, device, output_path)
