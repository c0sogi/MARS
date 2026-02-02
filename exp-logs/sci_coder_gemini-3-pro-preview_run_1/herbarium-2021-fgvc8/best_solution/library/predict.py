import torch
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import set_seed, get_device
from library.dataset import get_dataloaders
from library.model import HierarchicalEfficientNet


def predict(
    model_path=Config.MODEL_PATH, output_path=Config.SUBMISSION_PATH, debug=Config.DEBUG
):
    """
    Loads the trained model and generates predictions for the test dataset.

    Args:
        model_path (str): Path to the saved model checkpoint.
        output_path (str): Path where the submission CSV will be saved.
        debug (bool): If True, runs inference on a truncated dataset.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Running inference on device: {device}")

    # 2. Load Data
    # We need num_families to initialize the model architecture correctly.
    # get_dataloaders handles reading metadata and creating the test loader.
    print("Loading dataloaders...")
    _, _, test_loader, num_families = get_dataloaders(debug=debug)

    # 3. Initialize Model
    print(f"Initializing model with {num_families} auxiliary family classes...")
    model = HierarchicalEfficientNet(
        num_families=num_families,
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
    )

    # 4. Load Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # 5. Inference Loop
    all_preds = []
    all_ids = []

    print("Starting inference...")
    with torch.no_grad():
        for i, (images, image_ids) in enumerate(test_loader):
            images = images.to(device)

            # Forward pass: returns (species_logits, family_logits)
            # We only care about species prediction for submission
            species_logits, _ = model(images)

            # Get predicted class (argmax)
            preds = torch.argmax(species_logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_ids.append(image_ids.numpy())

    # 6. Generate Submission File
    all_preds = np.concatenate(all_preds)
    all_ids = np.concatenate(all_ids)

    print(f"Generated {len(all_preds)} predictions.")

    submission_df = pd.DataFrame({"Id": all_ids, "Predicted": all_preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
