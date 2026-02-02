import os
import torch
from library.config import Config, set_seed
from library.utils import get_device, load_checkpoint, logger
from library.vocab import Vocabulary
from library.model import GatedInfillingModel
from library.dataset import get_dataloaders
from library.engine import Engine


def generate_predictions(load_cached_data=True):
    """
    Loads the trained model and generates predictions for the test set.

    Args:
        load_cached_data (bool): Whether to use cached vocabulary and dataset files.
    """
    # 1. Setup
    set_seed()
    device = get_device()
    logger.info(f"Inference Device: {device}")

    # 2. Load Vocabulary
    vocab = Vocabulary()
    # This will load from Config.VOCAB_FILE if it exists and load_cached_data is True
    vocab.build_from_corpus(load_cached_data=load_cached_data)
    logger.info(f"Vocabulary loaded. Size: {len(vocab)}")

    # 3. Initialize Model
    # We must use the same architecture parameters as training
    model = GatedInfillingModel(
        vocab_size=len(vocab),
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        kernel_size=Config.KERNEL_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        padding_idx=vocab.pad_token_id,
    )
    model.to(device)

    # 4. Load Weights
    checkpoint_path = Config.MODEL_FILE
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Please run training (train.py) before generating predictions."
        )

    logger.info(f"Loading model checkpoint from {checkpoint_path}...")
    load_checkpoint(checkpoint_path, model, device=device)

    # 5. Prepare Data
    # get_dataloaders returns (train, val, test). We only need test.
    logger.info("Preparing test dataloader...")
    _, _, test_loader = get_dataloaders(vocab, load_cached_data=load_cached_data)

    # 6. Run Inference
    # We don't need optimizer or scheduler for inference
    engine = Engine(model, optimizer=None, scheduler=None, vocab=vocab, device=device)

    submission_path = Config.SUBMISSION_FILE

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    logger.info(f"Generating predictions for {len(test_loader.dataset)} samples...")
    engine.generate_submission(test_loader, submission_path)

    logger.info(f"Submission saved to {submission_path}")
