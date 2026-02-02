import os
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.model import HR1DResNet
from library.dataset import SmartphoneLocationDataset, collate_fn, generate_submission


def run_inference(
    checkpoint_path=None,
    output_path=None,
    batch_size=None,
    num_workers=None,
    device=None,
    load_cached=True,
):
    """
    Runs inference on the test set using the trained HR-1D-ResNet model and generates the submission file.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint. Defaults to Config.WORKING_DIR/best_model.pth.
        output_path (str, optional): Path to save the submission CSV. Defaults to Config.SUBMISSION_PATH.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to Config.BATCH_SIZE.
        num_workers (int, optional): Number of worker processes. Defaults to Config.NUM_WORKERS.
        device (str, optional): Computation device ('cpu' or 'cuda'). Defaults to Config.DEVICE.
        load_cached (bool, optional): Whether to use cached preprocessed data. Defaults to True.
    """
    # Set defaults if not provided
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    if device is None:
        device = Config.DEVICE

    print(f"--- Starting Inference ---")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")

    # 1. Initialize Model
    model = HR1DResNet()
    model.to(device)

    # 2. Load Weights
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}...")
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print(
            f"Warning: Checkpoint file not found at {checkpoint_path}. Inference will run with random weights."
        )

    # 3. Prepare Test Dataset
    print("Initializing Test Dataset...")
    try:
        # load_cached=True allows using precomputed parquet files if available
        test_dataset = SmartphoneLocationDataset(split="test", load_cached=load_cached)
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return

    if len(test_dataset) == 0:
        print("Test dataset is empty. Aborting inference.")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    # 4. Generate Submission
    # The generate_submission function in library.dataset handles the full inference loop,
    # including coordinate reconstruction (ENU -> ECEF -> WGS84) and merging with the sample submission.
    try:
        generate_submission(model, test_loader, output_path=output_path)
        print("Inference and submission generation completed successfully.")
    except Exception as e:
        print(f"Error during submission generation: {e}")
        raise e
