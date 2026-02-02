import os
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.tokenizer import TextProcessor
from library.dataset import StackExchangeDataset
from library.model import FastTextClassifier
from library.trainer import Trainer


def run_inference(load_cached_data: bool = True):
    """
    Orchestrates the inference process: loads resources, prepares data,
    initializes the model, and generates the submission file.

    Args:
        load_cached_data (bool): Whether to use cached parquet files for the dataset.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing inference on {device}...")

    # 2. Load Tokenizer and Mappings
    # We strictly load from cache to ensure consistency with the trained model.
    # If the tokenizer cache doesn't exist, inference cannot proceed correctly relative to the model weights.
    tokenizer = TextProcessor()
    tokenizer.fit(load_cached_data=True)

    # 3. Prepare Test Data
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(f"Test metadata file not found: {Config.TEST_METADATA}")

    test_dataset = StackExchangeDataset(
        metadata_path=Config.TEST_METADATA,
        tokenizer=tokenizer,
        split_name="test",
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=StackExchangeDataset.collate_fn,
        pin_memory=(device == "cuda"),
    )

    # 4. Initialize Model Architecture
    # We need to instantiate the model structure before loading weights.
    # Dimensions must match the tokenizer used during training.
    vocab_size = tokenizer.get_vocab_size()
    num_classes = tokenizer.get_num_tags()

    print(
        f"Initializing model with Vocab Size: {vocab_size}, Num Classes: {num_classes}"
    )

    model = FastTextClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
    )

    # 5. Initialize Trainer
    # We pass None for training-specific components (loaders, optimizer)
    # as they are not needed for the generate_submission method.
    trainer = Trainer(
        model=model,
        train_loader=None,
        val_loader=None,
        optimizer=None,
        tokenizer=tokenizer,
        device=device,
    )

    # 6. Generate Submission
    # This method handles loading the best state_dict and writing the CSV.
    trainer.generate_submission(
        test_loader=test_loader, output_path=Config.SUBMISSION_PATH
    )
