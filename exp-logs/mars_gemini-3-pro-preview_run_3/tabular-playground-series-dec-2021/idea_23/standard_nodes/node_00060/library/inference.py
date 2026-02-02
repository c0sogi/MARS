import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import DeepParallelDCNResNet
from library.data_utils import get_datasets


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across numpy and torch.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_predictions(
    load_cached_data=True, batch_size=None, num_workers=None, device_name=None
):
    """
    Loads the best model, generates predictions on the test set, and saves the submission file.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        batch_size (int, optional): Batch size for inference. Defaults to Config.BATCH_SIZE.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to Config.NUM_WORKERS.
        device_name (str, optional): Device to run inference on ('cpu' or 'cuda'). Defaults to Config.DEVICE.
    """
    # 1. Configuration and Setup
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS
    if device_name is None:
        device_name = Config.DEVICE

    set_seed(Config.SEED)
    device = torch.device(device_name)

    print(f"Starting inference on device: {device}")

    # 2. Load Data
    # We use get_datasets to ensure consistent preprocessing with training.
    # It returns train/val/test datasets, but we only need the test components here.
    _, _, test_dataset, test_ids, classes = get_datasets(
        load_cached_data=load_cached_data, debug=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device_name == "cuda" else False,
    )

    # 3. Initialize Model
    # Determine input dimension dynamically from the dataset
    # test_dataset[0] returns a tuple (features_tensor,), so we take shape of first element
    input_dim = test_dataset[0][0].shape[0]
    num_classes = len(classes)

    model = DeepParallelDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=Config.HIDDEN_DIM,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    # 4. Load Pre-trained Weights
    model_path = Config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Please train the model first."
        )

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 5. Inference Loop
    print("Generating predictions...")
    all_preds = []

    with torch.no_grad():
        for inputs in test_loader:
            # TensorDataset returns a tuple, unpack it
            inputs = inputs[0].to(device)

            outputs = model(inputs)

            # Get the class index with the highest probability
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())

    # 6. Post-processing
    # Map integer indices back to original class labels
    final_preds = classes[np.array(all_preds)]

    # 7. Save Submission
    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    submission = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: final_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions generated: {len(submission)}")
