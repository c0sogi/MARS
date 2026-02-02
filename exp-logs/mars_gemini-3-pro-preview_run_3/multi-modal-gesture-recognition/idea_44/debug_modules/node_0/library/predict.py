import os
import torch
import csv
from torch.utils.data import DataLoader
from library.config import Config
from library.model import RCMCN
from library.dataset import GestureDataset
from library.utils import decode_predictions_to_labels


def generate_submission(load_cached_data=True):
    """
    Generates the submission file for the test dataset.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # 1. Setup
    Config.set_seed(Config.SEED)
    Config.setup_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference using device: {device}")

    # 2. Load Test Data
    # We use transform=False because we don't want augmentation during inference
    test_dataset = GestureDataset(
        split="test", load_cached_data=load_cached_data, transform=False
    )

    # Batch size 1 is required because sequences have variable lengths
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    print(f"Loaded test dataset with {len(test_dataset)} samples.")

    # 3. Load Model
    model = RCMCN().to(device)

    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.BEST_MODEL_PATH}. Train the model first."
        )

    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded model from {Config.BEST_MODEL_PATH}")

    # 4. Inference Loop
    predictions = []

    with torch.no_grad():
        for i, (features, _) in enumerate(test_loader):
            features = features.to(device)

            # Retrieve sample_id from the dataset
            # The loader preserves order, and batch_size is 1
            sample_id = test_dataset.data[i]["sample_id"]

            # Forward pass
            # We only care about the final stage output (logits3)
            _, _, logits3 = model(features)

            # Convert to probabilities
            probs = torch.softmax(logits3, dim=2)

            # Decode: (1, T, C) -> (T, C) -> List of Labels
            frame_probs = probs.squeeze(0).cpu().numpy()
            pred_labels = decode_predictions_to_labels(frame_probs)

            predictions.append((sample_id, pred_labels))

    # 5. Write Submission File
    # Format: SessionID,Label1,Label2,...
    output_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Writing predictions to {output_path}...")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for sample_id, labels in predictions:
            # Row format: [sample_id, label1, label2, ...]
            row = [sample_id] + labels
            writer.writerow(row)

    print("Submission generation complete.")
