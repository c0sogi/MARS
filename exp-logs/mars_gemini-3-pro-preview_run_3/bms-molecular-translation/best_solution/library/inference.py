import os
import torch
from library.config import Config
from library.tokenizer import Tokenizer
from library.dataset import get_test_loader
from library.model import InChIModel
from library.trainer import Trainer
from library.utils import seed_everything


def run_inference(
    config: Config = None,
    debug: bool = False,
    debug_sample_size: int = 5000,
    load_cached_vocab: bool = True,
):
    """
    Runs the inference pipeline: loads the model, generates predictions for the test set,
    and saves the submission file.

    Args:
        config (Config, optional): Configuration object. If None, a new Config is instantiated.
        debug (bool): If True, runs in debug mode using a subset of data.
        debug_sample_size (int): Number of samples to use in debug mode.
        load_cached_vocab (bool): Whether to load the vocabulary from cache.
    """
    # 1. Setup Configuration
    if config is None:
        config = Config()

    # Apply overrides
    if debug:
        config.debug = True
        config.debug_sample_size = debug_sample_size

    # 2. Reproducibility
    seed_everything(config.seed)

    print(f"--- Initializing Inference (Debug={config.debug}) ---")

    # 3. Load Tokenizer
    # We need the vocabulary to initialize the model output layer correctly
    tokenizer = Tokenizer(config, load_cached_data=load_cached_vocab)
    print(f"Vocabulary size: {len(tokenizer)}")

    # 4. Initialize Model
    model = InChIModel(config, vocab_size=len(tokenizer))

    # 5. Prepare Test Loader
    test_loader = get_test_loader(config, tokenizer)
    print(f"Test loader initialized with {len(test_loader.dataset)} samples.")

    # 6. Initialize Trainer
    # We only provide the test_loader; train/val loaders are not needed for inference
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_loader=None,
        val_loader=None,
        test_loader=test_loader,
    )

    # 7. Run Prediction
    # This method handles loading the best checkpoint and saving the submission CSV
    trainer.predict()

    print("Inference completed successfully.")
