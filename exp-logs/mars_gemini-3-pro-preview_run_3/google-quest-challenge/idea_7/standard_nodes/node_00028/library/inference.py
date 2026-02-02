import os
import gc
import torch
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import (
    load_numpy_array,
    save_numpy_array,
    get_artifact_path,
)
from library.fine_tuning import FineTuningModel, extract_features as _extract_loop_core
from library.dataset import StackExchangeDataset


def extract_features(
    dataset,
    base_model_name,
    state_dict_path=None,
    batch_size=Config.VALID_BATCH_SIZE,
    device=None,
    load_cached_data=True,
    cache_name=None,
):
    """
    Extracts topological features [h_cls, h_q, h_a, h_diff] from a dataset using a
    fine-tuned backbone or a base model.

    This function handles model instantiation, weight loading, and the inference loop.
    It supports caching to avoid redundant computation.

    Args:
        dataset (StackExchangeDataset): The dataset to process.
        base_model_name (str): HuggingFace model name or path to a model directory
                               (e.g., the DAPT output path).
        state_dict_path (str, optional): Path to a .pth file containing the fine-tuned
                                         model weights (state_dict). If None, the weights
                                         from base_model_name are used.
        batch_size (int): Batch size for inference. Defaults to Config.VALID_BATCH_SIZE.
        device (torch.device, optional): The device to run inference on. If None,
                                         automatically detects CUDA/CPU.
        load_cached_data (bool): If True, attempts to load features from the cache
                                 directory before running inference.
        cache_name (str, optional): The filename (without extension) used for caching
                                    the output numpy array.

    Returns:
        np.ndarray: A numpy array of shape (N_samples, 4 * Hidden_Dim) containing
                    the concatenated features.
    """
    # 1. Check Cache
    # We strictly follow the logic: Try load -> If fail, compute -> Save
    if cache_name:
        cache_filename = f"{cache_name}.npy"
        if load_cached_data:
            cached_features = load_numpy_array(cache_filename)
            if cached_features is not None:
                print(f"Loaded cached features from {cache_filename}")
                return cached_features

    # 2. Setup Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Initializing model backbone: {base_model_name}")

    # Initialize the wrapper model
    # num_labels is required by the constructor for the linear head,
    # but strictly speaking we only need the backbone for feature extraction.
    # We pass 30 to match the training configuration.
    model = FineTuningModel(base_model_name, num_labels=30)

    # 3. Load Weights
    if state_dict_path:
        if os.path.exists(state_dict_path):
            print(f"Loading state dict from {state_dict_path}")
            state_dict = torch.load(state_dict_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: State dict path provided but file not found: {state_dict_path}"
            )
            print("Proceeding with base model weights (DAPT or Pretrained).")

    model.to(device)
    model.eval()

    # 4. Prepare DataLoader
    # We use num_workers=4 and pin_memory for efficient data throughput
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 5. Run Extraction Loop
    # We utilize the optimized loop provided in library.fine_tuning
    print(f"Starting feature extraction on {len(dataset)} samples...")
    features = _extract_loop_core(model, dataloader, device)

    # 6. Cache Results
    if cache_name:
        print(f"Caching features to {cache_name}.npy")
        save_numpy_array(features, f"{cache_name}.npy")

    # 7. Cleanup
    # Explicitly delete model and clear cache to free up VRAM for subsequent tasks
    del model
    del dataloader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return features
