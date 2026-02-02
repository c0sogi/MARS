import os
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.data_loader import get_data, GNSSDataset
from library.model import WindowedMLP, generate_submission


def run_inference(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    device=Config.DEVICE,
    checkpoint_path=Config.MODEL_CHECKPOINT_PATH,
    submission_path=Config.SUBMISSION_FILE_PATH,
    load_cached_data=True,
):
    """
    Executes the inference pipeline for the Smartphone Location Prediction task.

    Args:
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker processes for data loading.
        device (str): Device to run the model on ('cpu' or 'cuda').
        checkpoint_path (str): Path to the trained model weights.
        submission_path (str): Path where the submission CSV will be saved.
        load_cached_data (bool): Whether to attempt loading preprocessed data from cache.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    print("Starting Inference Pipeline...")

    # 1. Load Test Data
    # get_data handles caching and normalization.
    # For 'test' split, it returns (X, meta_list, df_original).
    # It assumes scaler_stats.json exists (generated during training).
    try:
        X_test, meta_test, df_test_original = get_data(
            split="test", load_cached_data=load_cached_data
        )
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Ensure training has been run to generate scaler statistics.")
        raise

    print(f"Test data loaded. Total samples: {len(X_test)}")

    # 2. Create Dataset and DataLoader
    test_dataset = GNSSDataset(X_test, mode="test")

    use_pin_memory = device == "cuda"

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    # 3. Initialize Model
    # The architecture must match the one used during training.
    model = WindowedMLP(
        input_dim=Config.INPUT_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        output_dim=Config.OUTPUT_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # 4. Load Model Weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. Please train the model first."
        )

    print(f"Loading model weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5. Generate Submission
    # generate_submission handles the prediction loop, reconstruction of Lat/Lon from residuals,
    # merging with the sample submission template, and saving to CSV.
    df_submission = generate_submission(
        model=model,
        test_loader=test_loader,
        meta_list=meta_test,
        df_test_original=df_test_original,
        submission_path=submission_path,
        device=device,
    )

    print("Inference Pipeline Completed Successfully.")
    return df_submission
