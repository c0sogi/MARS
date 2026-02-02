import os
import torch
from library.config import Config
from library.model import ViT2InChI
from library.dataset import get_dataloaders
from library.tokenizer import InChITokenizer
from library.utils import load_checkpoint
from library.trainer import predict_and_submit


def greedy_decode(model, images, max_len=Config.MAX_TEXT_LEN, device=Config.DEVICE):
    """
    Performs greedy decoding to generate InChI sequences from images.

    Args:
        model (ViT2InChI): The trained model.
        images (torch.Tensor): Batch of input images (B, C, H, W).
        max_len (int): Maximum length of the generated sequence.
        device (str): Device to run inference on.

    Returns:
        torch.Tensor: Generated token indices (B, seq_len).
    """
    # The model's predict method implements the encoding and iterative decoding logic
    return model.predict(images, max_len=max_len, device=device)


def generate_submission(
    checkpoint_path=Config.BEST_MODEL_PATH,
    output_path="./submission/submission.csv",
    debug=False,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Runs the full inference pipeline on the test set and generates a submission file.

    Args:
        checkpoint_path (str): Path to the trained model checkpoint.
        output_path (str): Path to save the submission CSV.
        debug (bool): If True, runs on a subset of the data.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker processes for data loading.
        load_cached_data (bool): Whether to use cached metadata.
    """
    # Setup
    device = Config.DEVICE
    tokenizer = InChITokenizer()

    print(f"Initializing inference on device: {device}")

    # Load Data
    # get_dataloaders returns (train, val, test). We only need test.
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        load_cached_data=load_cached_data,
        debug=debug,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Initialize Model
    print("Initializing model...")
    model = ViT2InChI().to(device)

    # Load Weights
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        # load_checkpoint loads weights in-place
        load_checkpoint(checkpoint_path, model, device=device)
    else:
        print(
            f"WARNING: Checkpoint not found at {checkpoint_path}. Using random weights."
        )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate Predictions and Save
    # predict_and_submit handles the loop over the dataloader, decoding, and CSV saving
    predict_and_submit(
        model=model,
        test_loader=test_loader,
        device=device,
        tokenizer=tokenizer,
        save_path=output_path,
    )

    print(f"Inference complete. Submission saved to {output_path}")
