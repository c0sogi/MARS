import os
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.vocab_manager import VocabManager
from library.window_processor import WindowProcessor
from library.data_loader import WindowNQDataset
from library.model import WindowMaxPoolingNetwork
from library.solver import Solver


def run_inference_pipeline(
    config_class=Config,
    load_cached_data: bool = True,
    device_name: str = None,
    debug: bool = False,
    debug_size: int = 2000,
):
    """
    Executes the inference pipeline for the Window-Based Max-Pooling Network.

    Args:
        config_class: The configuration class to use.
        load_cached_data (bool): Whether to try loading processed features from cache.
        device_name (str): 'cpu' or 'cuda'. If None, detects automatically.
        debug (bool): If True, runs on a small subset of the test data.
        debug_size (int): Number of samples to use if debug is True.
    """
    # 1. Configuration Setup
    config = config_class()
    config.DEBUG = debug
    config.DEBUG_SIZE = debug_size
    config.setup()

    # 2. Device Selection
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Initializing Inference Pipeline on {device}...")

    # 3. Vocabulary and Embeddings
    # We assume the vocabulary has been built during the training phase.
    # We force load_cached_data=True for vocab to ensure consistency with the trained model.
    vocab_manager = VocabManager(config)
    vocab_manager.build_vocab(load_cached_data=True)

    embedding_matrix = vocab_manager.get_embedding_matrix()

    # 4. Data Processing and Loading
    # We use WindowProcessor directly to avoid loading training data overhead
    print("Processing test data...")
    processor = WindowProcessor(config, vocab_manager)

    # Process dataset (handles caching logic internally via to_parquet/read_parquet)
    test_features = processor.process_dataset(
        load_cached_data=load_cached_data, is_train=False
    )

    # Create Dataset
    test_dataset = WindowNQDataset(test_features, split="test", config=config)

    # Create DataLoader
    num_workers = 2 if os.cpu_count() > 2 else 0
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE * 2,  # Double batch size for inference
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"Test Data Loaded: {len(test_dataset)} windows from {len(test_features['example_id'].unique())} examples."
    )

    # 5. Model Initialization
    print("Initializing model architecture...")
    model = WindowMaxPoolingNetwork(embedding_matrix, config)

    # 6. Solver Initialization and Inference
    # Solver handles checkpoint loading, forward pass, aggregation, and submission generation
    solver = Solver(model, config, device=device)

    print("Starting prediction generation...")
    solver.inference(test_loader)

    print("Inference pipeline completed successfully.")
