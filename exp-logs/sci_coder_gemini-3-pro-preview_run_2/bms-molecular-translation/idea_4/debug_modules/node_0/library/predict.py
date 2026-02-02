import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.model import ShowAttendTell
from library.tokenizer import Tokenizer
from library.dataset import get_test_dataloader


def generate_predictions(debug=False, load_cached_data=True):
    """
    Generates predictions for the test dataset and saves them to a CSV file.

    Args:
        debug (bool): If True, runs on a small subset of the test data.
        load_cached_data (bool): If True, attempts to load metadata from cache.
    """
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    # We assume the tokenizer has already been built during training.
    # If not, it will try to build from train metadata.
    tokenizer = Tokenizer(load_cached_data=load_cached_data)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    # 2. Load Test Data
    test_loader = get_test_dataloader(
        tokenizer,
        batch_size=Config.BATCH_SIZE,
        debug=debug,
        load_cached_data=load_cached_data,
    )
    print(f"Test data loaded. Number of batches: {len(test_loader)}")

    # 3. Initialize Model
    model = ShowAttendTell(vocab_size=vocab_size).to(device)

    # 4. Load Checkpoint
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}. Please train the model first."
        )

    load_checkpoint(Config.MODEL_SAVE_PATH, model, device=Config.DEVICE)

    # 5. Inference Loop
    model.eval()
    predictions = []

    print("Starting inference...")

    with torch.no_grad():
        for i, (images, image_ids) in enumerate(test_loader):
            images = images.to(device)

            # Forward pass in inference mode (captions=None triggers greedy decoding)
            # Output shape: (batch_size, max_len, vocab_size)
            outputs = model(images, captions=None)

            # Get predicted indices: (batch_size, max_len)
            predicted_indices = torch.argmax(outputs, dim=2)

            # Convert indices to text
            for idx in range(images.size(0)):
                seq = predicted_indices[idx].cpu().tolist()
                inchi_string = tokenizer.sequence_to_text(seq)

                predictions.append({"image_id": image_ids[idx], "InChI": inchi_string})

            if i % 50 == 0:
                print(f"Processed batch {i}/{len(test_loader)}")

    # 6. Save Submission
    df_submission = pd.DataFrame(predictions)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    submission_path = Config.SUBMISSION_PATH
    df_submission.to_csv(submission_path, index=False)

    print(f"Inference complete. Submission saved to {submission_path}")
    print(f"Total predictions generated: {len(df_submission)}")
    print("First 5 predictions:")
    print(df_submission.head())


if __name__ == "__main__":
    # Example usage (not executed when imported)
    generate_predictions(debug=Config.DEBUG, load_cached_data=True)
