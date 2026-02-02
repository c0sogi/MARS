import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.tokenizer import Tokenizer
from library.dataset import InChiDataset, get_transforms, collate_fn
from library.model import VisualTransformer
from library.train import inference, seed_everything


def generate_submission(debug: bool = False):
    """
    Orchestrates the inference process: loads data and model, generates predictions,
    and saves the submission file.

    Args:
        debug (bool): If True, runs inference on a small subset of the test data.
    """
    # 1. Setup Environment
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("--- Starting Prediction Process ---")

    # 2. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_METADATA}")

    df_test = pd.read_csv(Config.TEST_METADATA)

    if debug:
        print("Debug mode: Sampling test data (n=100).")
        df_test = df_test.sample(n=100, random_state=Config.SEED).reset_index(drop=True)

    print(f"Test dataset size: {len(df_test)}")

    # 3. Initialize Tokenizer
    # We expect the tokenizer vocabulary to have been built during training (cached).
    tokenizer = Tokenizer(load_cached_data=True, debug=debug)

    # 4. Prepare Dataset and DataLoader
    test_dataset = InChiDataset(df_test, tokenizer, transform=get_transforms("test"))

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 5. Initialize Model
    print("Initializing model architecture...")
    model = VisualTransformer(vocab_size=len(tokenizer))
    model = model.to(device)

    # 6. Load Model Weights
    if os.path.exists(Config.MODEL_PATH):
        print(f"Loading model weights from {Config.MODEL_PATH}...")
        checkpoint = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.MODEL_PATH}. Using random weights."
        )

    # 7. Run Inference
    # Uses the inference function from library.train which performs greedy decoding
    ids, preds = inference(test_loader, model, tokenizer, device)

    # 8. Create and Save Submission
    sub_df = pd.DataFrame({"image_id": ids, "InChI": preds})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(f"Submission shape: {sub_df.shape}")
    print("Head of submission:")
    print(sub_df.head())

    return sub_df
