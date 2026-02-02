import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.model import PlantClassifier
from library.dataset import create_dataloaders
from library.utils import set_seed


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).
    Averages softmax probabilities from the original and flipped images.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): The test data loader.
        device (torch.device): The computation device.

    Returns:
        tuple: (list of image_ids, list of predicted_labels)
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # Forward pass 1: Original image
            logits_1 = model(images)
            probs_1 = F.softmax(logits_1, dim=1)

            # Forward pass 2: Horizontally flipped image
            images_flip = torch.flip(images, dims=[3])
            logits_2 = model(images_flip)
            probs_2 = F.softmax(logits_2, dim=1)

            # Average probabilities
            avg_probs = (probs_1 + probs_2) / 2.0

            # Get predicted class index
            preds = torch.argmax(avg_probs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_ids.extend(ids)

    return all_ids, all_preds


def generate_submission(
    model_path="./working/idea_1/best_model.pth",
    output_dir="./submission",
    batch_size=64,
    num_workers=4,
    debug=False,
    img_size=260,
):
    """
    Runs the inference pipeline and generates the submission file.

    Args:
        model_path (str): Path to the trained model weights.
        output_dir (str): Directory to save the submission file.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        debug (bool): If True, runs on a small subset of data.
        img_size (int): Image resolution (must match training).
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting inference on device: {device}")

    # 1. Load Data
    # create_dataloaders returns (train, val, test). We only need test.
    _, _, test_loader = create_dataloaders(
        train_batch_size=batch_size,  # Unused
        val_batch_size=batch_size,
        num_workers=num_workers,
        debug=debug,
        img_size=img_size,
    )

    # 2. Initialize Model
    # num_classes must match the training configuration (15501)
    model = PlantClassifier(
        num_classes=15501, model_name="tf_efficientnetv2_s", pretrained=False
    )

    # 3. Load Weights
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model file not found at {model_path}. Using random weights (expect poor performance)."
        )

    model.to(device)

    # 4. Generate Predictions
    print("Generating predictions with TTA...")
    ids, preds = predict_with_tta(model, test_loader, device)

    # 5. Format and Save Submission
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({"Id": ids, "Predicted": preds})

    # Ensure Id is integer for correct sorting (dataset returns strings)
    df["Id"] = df["Id"].astype(int)
    df = df.sort_values("Id").reset_index(drop=True)

    output_path = os.path.join(output_dir, "submission.csv")
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
