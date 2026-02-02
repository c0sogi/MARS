import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import (
    TRAIN_CSV,
    TEST_CSV,
    TRAIN_CACHE_PATH,
    TEST_CACHE_PATH,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE,
    ATOM_FEATURES_DIM,
    GLOBAL_FEATURES_DIM,
    HIDDEN_DIM,
    ATOMIC_LAYERS,
    GLOBAL_LAYERS,
    FUSION_LAYERS,
    DROPOUT,
    USE_BATCH_NORM,
    SEED,
)
from library.data import process_data, get_scalers, CrystalDataset, collate_sparse
from library.model import REMSWDSModel
from library.train import predict

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def generate_predictions(
    load_cached_data: bool = True, batch_size: int = BATCH_SIZE, device_name: str = None
):
    """
    Generates predictions for the test set using the trained REMS-WDS model.

    Args:
        load_cached_data (bool): Whether to load pre-computed features from cache.
        batch_size (int): Batch size for inference.
        device_name (str): Device to use ('cuda' or 'cpu'). If None, auto-detects.
    """
    # 1. Setup Device
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"Using device: {device}")

    # 2. Load Metadata
    print("Loading metadata...")
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        raise FileNotFoundError(
            "Metadata CSV files not found. Ensure metadata generation was successful."
        )

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # 3. Process Data
    # We need to process training data to fit the scalers correctly
    print("Processing training data for scaler fitting...")
    train_af, train_gf, _, _ = process_data(
        train_df, TRAIN_CACHE_PATH, load_cached_data=load_cached_data
    )

    print("Processing test data...")
    test_af, test_gf, test_y, test_ids = process_data(
        test_df, TEST_CACHE_PATH, load_cached_data=load_cached_data
    )

    # 4. Fit Scalers
    # Scalers must be fitted on training distribution
    print("Fitting scalers...")
    scaler_atomic, scaler_global = get_scalers(train_af, train_gf)

    # 5. Create Test Dataset and Loader
    print("Creating test dataset...")
    test_dataset = CrystalDataset(
        test_af,
        test_gf,
        test_y,  # These are likely NaNs or placeholders
        test_ids,
        scaler_atomic=scaler_atomic,
        scaler_global=scaler_global,
        mode="test",  # Important: this skips the log1p transform on targets (though targets are ignored for inference)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sparse,
        num_workers=0,
    )

    # 6. Initialize Model
    print("Initializing model architecture...")
    model = REMSWDSModel(
        atom_features_dim=ATOM_FEATURES_DIM,
        global_features_dim=GLOBAL_FEATURES_DIM,
        hidden_dim=HIDDEN_DIM,
        atomic_layers=ATOMIC_LAYERS,
        global_layers=GLOBAL_LAYERS,
        fusion_layers=FUSION_LAYERS,
        dropout=DROPOUT,
        use_bn=USE_BATCH_NORM,
    ).to(device)

    # 7. Load Trained Weights
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_SAVE_PATH}. Train the model first."
        )

    print(f"Loading model weights from {MODEL_SAVE_PATH}...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    # 8. Run Inference
    print("Running inference...")
    # predict() function from library.train handles evaluation loop and inverse transform (expm1)
    predictions, ids = predict(model, test_loader, device)

    # 9. Create Submission File
    print("Formatting submission...")
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Save
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print("Sample predictions:")
    print(submission_df.head())


if __name__ == "__main__":
    generate_predictions()
