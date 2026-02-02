import os
import torch
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast

from library.config import Config
from library.model import get_model
from library.dataset import get_dataloaders, get_label_mapping


def predict_and_submit(debug=False, model_path=None):
    """
    Loads the trained model, performs inference on the test set, and generates
    the submission CSV file.

    Args:
        debug (bool): If True, runs inference on a small subset of the test data.
        model_path (str, optional): Path to the trained model weights.
                                    Defaults to Config.IDEA_DIR/model.pth.
    """
    # 1. Setup Device
    device = torch.device(Config.DEVICE)
    print(f"Running inference on device: {device}")

    # 2. Prepare DataLoaders
    # We only need the test_loader, but get_dataloaders returns all three.
    print(f"Loading dataloaders (Debug={debug})...")
    _, _, test_loader = get_dataloaders(debug=debug)

    # 3. Load Label Mapping
    # We need to map model outputs (0..N-1) back to category_ids.
    # This mapping is created during training/dataset initialization.
    classes_npy_path = os.path.join(Config.IDEA_DIR, "classes.npy")

    if os.path.exists(classes_npy_path):
        unique_cats = np.load(classes_npy_path)
        print(f"Loaded class mapping from {classes_npy_path}")
    else:
        print("Class mapping cache not found. Regenerating from training metadata...")
        train_df = pd.read_csv(Config.TRAIN_CSV)
        _, unique_cats = get_label_mapping(train_df, load_cached_data=False)

    # 4. Initialize Model
    print("Initializing model...")
    # We use pretrained=False here because we are about to load our own weights.
    # The architecture remains the same.
    model = get_model(pretrained=False)
    model.to(device)

    # 5. Load Weights
    if model_path is None:
        model_path = os.path.join(Config.IDEA_DIR, "model.pth")

    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {model_path}. Using random weights (expect poor performance)."
        )

    # 6. Inference Loop
    model.eval()
    all_preds = []
    all_ids = []

    print("Starting inference loop...")
    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            # Use Mixed Precision if available/configured
            with autocast():
                outputs = model(images)

            # Get predicted class indices (0..N-1)
            preds_indices = torch.argmax(outputs, dim=1).cpu().numpy()

            # Map indices back to original category_ids
            mapped_preds = unique_cats[preds_indices]

            all_preds.extend(mapped_preds)
            all_ids.extend(image_ids.numpy())

    # 7. Generate Submission
    print(f"Generating submission file with {len(all_preds)} predictions...")
    submission_df = pd.DataFrame({"Id": all_ids, "Predicted": all_preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
