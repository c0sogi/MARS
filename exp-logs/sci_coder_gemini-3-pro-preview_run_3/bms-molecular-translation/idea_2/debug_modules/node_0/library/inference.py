import os
import torch
import pandas as pd
from library.config import Config
from library.tokenizer import Tokenizer
from library.dataset import get_dataloaders
from library.model import InChiModel
from library.utils import load_checkpoint


def generate_submission(load_cached_data: bool = True, debug: bool = False):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        load_cached_data (bool): Whether to load the tokenizer vocabulary from cache.
        debug (bool): If True, runs inference on a small subset of the test data.
    """
    print("--- Initializing Inference ---")

    # 1. Setup Tokenizer
    tokenizer = Tokenizer(load_cached_data=load_cached_data)

    # 2. Setup Test DataLoader
    # We only need the test_loader. The get_dataloaders function returns (train, val, test).
    subset_size = 100 if debug else None
    _, _, test_loader = get_dataloaders(
        tokenizer=tokenizer,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug_subset_size=subset_size,
    )

    # 3. Setup Model
    vocab_size = len(tokenizer)
    model = InChiModel(vocab_size=vocab_size)
    model.to(Config.DEVICE)

    # 4. Load Checkpoint
    checkpoint_path = os.path.join(
        Config.WORKING_DIR, "checkpoints", "model_best.pth.tar"
    )
    if not os.path.exists(checkpoint_path):
        print(
            f"Warning: Best model checkpoint not found at {checkpoint_path}. Checking for latest checkpoint..."
        )
        checkpoint_path = os.path.join(
            Config.WORKING_DIR, "checkpoints", "checkpoint.pth.tar"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No checkpoints found in {os.path.join(Config.WORKING_DIR, 'checkpoints')}"
            )

    load_checkpoint(checkpoint_path, model)
    model.eval()

    # 5. Inference Loop
    predictions = []

    # Access the underlying dataframe to get image_ids
    # The test_loader dataset is an instance of InChiDataset
    test_df = test_loader.dataset.df

    print(f"Starting inference on {len(test_df)} samples...")

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(Config.DEVICE)

            # Greedy decoding using the model's predict method
            # Output shape: (Batch_Size, Seq_Len)
            pred_seqs = model.predict(images, tokenizer, max_len=Config.MAX_LEN)

            # Convert integer sequences to InChI strings
            for seq in pred_seqs:
                inchi_str = tokenizer.sequence_to_text(seq)
                predictions.append(inchi_str)

            if i % 10 == 0:
                print(f"Processed batch {i}/{len(test_loader)}")

    # 6. Align Predictions and Save
    # Ensure prediction count matches metadata (handling potential drop_last=False logic correctly)
    if len(predictions) != len(test_df):
        print(
            f"Note: Prediction count ({len(predictions)}) differs from metadata count ({len(test_df)})."
        )
        # If debug mode is on, test_df is already subsetted, but dataloader might have dropped last if configured (it's not)
        # Just ensure length alignment for safety
        min_len = min(len(predictions), len(test_df))
        predictions = predictions[:min_len]
        test_df = test_df.iloc[:min_len]

    submission_df = pd.DataFrame(
        {"image_id": test_df["image_id"].values, "InChI": predictions}
    )

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission generation complete.")
