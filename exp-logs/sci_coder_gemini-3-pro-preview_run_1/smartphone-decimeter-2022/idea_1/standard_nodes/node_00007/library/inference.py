import os
import torch
import pandas as pd
from library.data_loader import GnssWindowedDataset
from library.model import TemporalConvNet, generate_submission


def generate_predictions(
    input_dir="./input",
    metadata_dir="./metadata",
    working_dir="./working",
    submission_dir="./submission",
    config=None,
):
    """
    Manages the prediction pipeline for the test set.
    Restores the trained model and scaler, then generates predictions.

    Args:
        input_dir (str): Path to input data.
        metadata_dir (str): Path to metadata files.
        working_dir (str): Path to working directory (model weights).
        submission_dir (str): Path to save submission file.
        config (dict): Hyperparameters.
    """
    # Default configuration
    if config is None:
        config = {
            "window_size": 64,
            "batch_size": 256,
            "hidden_dim": 128,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

    # Ensure working directory for cache exists
    # The data loader uses ./working/idea_3/ by default
    cache_dir = os.path.join(working_dir, "idea_3")
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Restore Scaler
    # We need to fit the scaler on training data exactly as done during training.
    # This ensures test data is normalized using the same statistics.
    print("Restoring scaler from training data...")
    train_meta_path = os.path.join(metadata_dir, "train_metadata.csv")
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Train metadata not found at {train_meta_path}")

    df_train = pd.read_csv(train_meta_path)

    # Initialize dataset in train mode to fit scaler
    # The dataset class handles caching of preprocessed files
    train_dataset = GnssWindowedDataset(
        metadata_df=df_train,
        input_dir=input_dir,
        window_size=config["window_size"],
        mode="train",
        scaler=None,
    )
    scaler = train_dataset.scaler
    print("Scaler restored.")

    # 2. Restore Model
    print("Restoring model...")
    # Input channels fixed at 5 based on data_loader features:
    # [WlsAlt, Cn0DbHz, SvElevationDegrees, SatCount, RawPseudorangeUncertaintyMeters]
    model = TemporalConvNet(
        input_channels=5,
        window_size=config["window_size"],
        hidden_dim=config["hidden_dim"],
        output_dim=2,
    )

    weights_path = os.path.join(working_dir, "model_weights.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. Train model first."
        )

    model.load_state_dict(torch.load(weights_path, map_location=config["device"]))
    model.to(config["device"])
    model.eval()
    print("Model loaded.")

    # 3. Generate Submission
    test_meta_path = os.path.join(metadata_dir, "test_metadata.csv")
    output_file = os.path.join(submission_dir, "submission.csv")

    # Ensure submission dir exists
    os.makedirs(submission_dir, exist_ok=True)

    # generate_submission handles loading test metadata, creating the test dataset,
    # running inference, reconstructing coordinates, and saving the CSV.
    generate_submission(
        model=model,
        test_metadata_path=test_meta_path,
        input_dir=input_dir,
        output_file=output_file,
        config=config,
        scaler=scaler,
    )
    print("Inference complete.")
