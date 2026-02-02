import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import BirdDataset, get_transforms, load_dataframe
from library.model import BirdClassifier
from library.utils import set_seed


def generate_predictions(
    model_path: str = Config.MODEL_SAVE_PATH,
    output_path: str = Config.SUBMISSION_PATH,
    batch_size: int = Config.BATCH_SIZE,
    debug: bool = Config.DEBUG,
    device: torch.device = Config.DEVICE,
):
    """
    Generates predictions for the test set and saves them in the submission format.

    Args:
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        debug (bool): Whether to run in debug mode (subset of data).
        device (torch.device): Device to run inference on.
    """
    set_seed(Config.SEED)

    print(f"Starting inference on device: {device}")

    # --- 1. Load Data ---
    # Load test metadata
    df_test = load_dataframe(Config.TEST_CSV, debug=debug)

    # Initialize Dataset and DataLoader
    test_dataset = BirdDataset(
        df=df_test,
        transforms=get_transforms("test"),
        img_dir=Config.FILTERED_SPECTROGRAM_DIR,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 2. Load Model ---
    model = BirdClassifier(
        model_name=Config.MODEL_NAME,
        pretrained=False,  # No need to download pretrained weights again, we load state_dict
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        # Handle case where checkpoint saves full dict vs just state_dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model.to(device)
    model.eval()

    # --- 3. Inference ---
    results = []

    with torch.no_grad():
        for images, _, rec_ids in test_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Move to CPU and convert to numpy
            probs = probs.cpu().numpy()
            rec_ids = (
                rec_ids.numpy() if not isinstance(rec_ids, np.ndarray) else rec_ids
            )

            # Store results
            for i in range(len(rec_ids)):
                results.append({"rec_id": rec_ids[i], "probs": probs[i]})

    # --- 4. Format Submission ---
    submission_rows = []

    for item in results:
        r_id = int(item["rec_id"])
        probabilities = item["probs"]

        # Iterate over all 19 species
        for species_idx in range(Config.NUM_CLASSES):
            # Construct the submission Id: rec_id * 100 + species_number
            submission_id = r_id * 100 + species_idx
            prob = probabilities[species_idx]

            submission_rows.append([submission_id, prob])

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows, columns=["Id", "Probability"])

    # Sort by Id for cleanliness (though not strictly required)
    submission_df = submission_df.sort_values("Id").reset_index(drop=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions generated: {len(submission_df)}")
