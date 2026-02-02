import os
import torch
import pandas as pd
import numpy as np
import scipy.signal
from itertools import groupby
from library.config import Config
from library.model import BiGRUModel
from library.data_loader import get_processed_data
from library.utils import setup_logger, set_seeds


def load_trained_model(model_path, device):
    """
    Instantiates the model and loads weights from the checkpoint.
    """
    model = BiGRUModel(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
    )

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model.to(device)
    model.eval()
    return model


def predict_sequence(model, features, device):
    """
    Runs inference on a single sequence of features.
    Args:
        model: Trained PyTorch model.
        features: Numpy array of shape (SeqLen, InputDim).
        device: Torch device.
    Returns:
        np.array: Frame-wise class predictions of shape (SeqLen,).
    """
    # Prepare input tensor
    # Model expects (Batch, SeqLen, InputDim)
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)

    # Lengths tensor for packing (Batch,)
    lengths = torch.tensor([len(features)], dtype=torch.long)

    with torch.no_grad():
        # Forward pass
        logits = model(features_tensor, lengths)
        # logits: (1, SeqLen, NumClasses)

        # Get probabilities or just argmax
        # We just need argmax for hard label prediction
        predictions = torch.argmax(logits, dim=2)  # (1, SeqLen)

    return predictions.squeeze(0).cpu().numpy()


def post_process_predictions(raw_predictions):
    """
    Applies smoothing and filtering to raw frame predictions.
    1. Median Filter.
    2. Group consecutive identical labels.
    3. Filter background and short segments.

    Args:
        raw_predictions (np.array): 1D array of class IDs.

    Returns:
        list: Ordered list of recognized gesture IDs (strings).
    """
    # 1. Median Filtering to remove noise
    # kernel_size must be odd
    k_size = Config.SMOOTHING_KERNEL_SIZE
    if k_size % 2 == 0:
        k_size += 1

    smoothed = scipy.signal.medfilt(raw_predictions, kernel_size=k_size)

    final_gestures = []

    # 2. Group consecutive labels
    for label, group in groupby(smoothed):
        label = int(label)
        length = sum(1 for _ in group)

        # 3. Filter
        # Must not be background
        if label == Config.BACKGROUND_CLASS_ID:
            continue

        # Must be long enough
        if length < Config.MIN_GESTURE_LENGTH:
            continue

        final_gestures.append(str(label))

    return final_gestures


def run_inference(load_cached_data=True):
    """
    Main function to run the inference pipeline.
    """
    logger = setup_logger("Inference")
    set_seeds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running inference on device: {device}")

    # 1. Load Data
    # We need sample IDs to match predictions to files.
    # get_processed_data returns features aligned with the metadata CSV rows.
    if not os.path.exists(Config.TEST_METADATA_PATH):
        logger.error(f"Test metadata not found at {Config.TEST_METADATA_PATH}")
        return

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    sample_ids = test_df["sample_id"].tolist()

    logger.info("Loading test data...")
    features_list, _ = get_processed_data(
        Config.TEST_METADATA_PATH,
        Config.TEST_CACHE_PATH,
        load_cached_data=load_cached_data,
    )

    if len(features_list) != len(sample_ids):
        logger.error("Mismatch between loaded features and metadata rows.")
        return

    # 2. Load Model
    try:
        model = load_trained_model(Config.MODEL_SAVE_PATH, device)
        logger.info(f"Model loaded from {Config.MODEL_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 3. Predict and Generate Submission
    results = []
    logger.info("Starting prediction loop...")

    for i, (sid, feats) in enumerate(zip(sample_ids, features_list)):
        if feats is None or len(feats) == 0:
            # Handle empty/invalid features
            prediction_str = ""
        else:
            # Predict
            raw_preds = predict_sequence(model, feats, device)

            # Post-process
            gesture_list = post_process_predictions(raw_preds)

            # Join with spaces
            prediction_str = " ".join(gesture_list)

        # Extract Integer ID from SampleXXXXX
        try:
            seq_id = int(sid.replace("Sample", ""))
        except ValueError:
            seq_id = sid

        # Format: Id,Sequence
        line = f"{seq_id},{prediction_str}"
        results.append(line)

        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(sample_ids)} samples.")

    # 4. Save Submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    # The submission format requires specific columns or just the lines?
    # The prompt says: "Session00001,2,12,3"
    # Usually, a header is not strictly required if not specified,
    # but standard CSV usually has one. The prompt example doesn't show a header.
    # However, to be safe, we will write lines directly.

    try:
        with open(Config.SUBMISSION_FILE, "w") as f:
            f.write("Id,Sequence\n")
            for line in results:
                f.write(line + "\n")
        logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
    except Exception as e:
        logger.error(f"Failed to write submission file: {e}")


if __name__ == "__main__":
    # This block is for testing the module independently if needed
    run_inference()
