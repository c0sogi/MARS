import os
import torch
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import CNNTransformer
from library.tokenizer import InChITokenizer


def run_inference(
    checkpoint_path: str = Config.MODEL_SAVE_PATH,
    output_path: str = Config.SUBMISSION_PATH,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    debug: bool = Config.DEBUG,
    debug_sample_size: int = Config.DEBUG_SAMPLE_SIZE,
):
    """
    Executes the inference pipeline: loads data, loads model, generates predictions,
    and saves the submission file.

    Args:
        checkpoint_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        device (str): Computation device ('cuda' or 'cpu').
        debug (bool): Whether to run in debug mode (subset of data).
        debug_sample_size (int): Number of samples to use in debug mode.
    """
    # 1. Set seed for reproducibility
    seed_everything(Config.SEED)
    print(f"Starting inference on device: {device}")

    # 2. Prepare DataLoaders
    # We only need the test_loader for inference
    _, _, test_loader = get_dataloaders(
        test_metadata_path=Config.TEST_METADATA_PATH,
        batch_size=batch_size,
        num_workers=num_workers,
        debug=debug,
        debug_sample_size=debug_sample_size,
    )

    if test_loader is None:
        print(f"Error: Could not load test data from {Config.TEST_METADATA_PATH}")
        return

    # 3. Initialize Model
    model = CNNTransformer().to(device)

    # 4. Load Checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}...")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # Support both full checkpoint dict and direct state_dict
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(
            f"Warning: Checkpoint file not found at {checkpoint_path}. Using random initialization."
        )

    model.eval()

    # 5. Initialize Tokenizer
    tokenizer = InChITokenizer()

    # 6. Inference Loop
    image_ids = []
    inchi_preds = []

    print(f"Processing {len(test_loader.dataset)} samples...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["images"].to(device)
            ids = batch["image_ids"]

            # Perform greedy decoding using the model's predict method
            # This handles the autoregressive generation loop internally
            pred_indices = model.predict(images, max_len=Config.MAX_LEN, device=device)

            # Decode indices to strings
            for k in range(len(pred_indices)):
                # Convert tensor indices to list
                indices_list = pred_indices[k]
                # Decode to string (handles special tokens like SOS, EOS, PAD)
                pred_str = tokenizer.decode(indices_list)

                image_ids.append(ids[k])
                inchi_preds.append(pred_str)

    # 7. Generate Submission File
    df_submission = pd.DataFrame({"image_id": image_ids, "InChI": inchi_preds})

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_submission.to_csv(output_path, index=False)

    print(f"Inference complete. Submission saved to {output_path}")
    print(f"Total predictions: {len(df_submission)}")

    if not df_submission.empty:
        print("Sample predictions:")
        print(df_submission.head())

    return df_submission
