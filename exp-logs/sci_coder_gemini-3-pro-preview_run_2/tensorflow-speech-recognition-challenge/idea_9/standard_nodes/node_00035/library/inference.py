import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, get_device
from library.dataset import get_dataloaders, IDX2LABEL
from library.model import get_model


def generate_submission(debug: bool = False):
    """
    Loads the best trained model, runs inference on the test dataset,
    and generates a submission CSV file.

    Args:
        debug (bool): If True, runs inference on a small subset of the test data.
    """
    # 1. Setup
    set_seed(Config.seed)
    device = get_device()

    print(f"Initializing inference (Debug={debug})...")

    # 2. Load Data
    # We only need the test_loader
    _, _, test_loader = get_dataloaders(debug=debug)

    # 3. Initialize Model
    model = get_model()
    model.to(device)

    # 4. Load Weights
    checkpoint_path = Config.best_model_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Best model checkpoint not found at: {checkpoint_path}. "
            "Please ensure training has completed successfully."
        )

    print(f"Loading model weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle checkpoint structure (state_dict vs full checkpoint dict)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        best_acc = checkpoint.get("best_acc", "N/A")
        print(f"Model loaded. Training Best Accuracy: {best_acc}")
    else:
        model.load_state_dict(checkpoint)
        print("Model loaded (raw state_dict).")

    model.eval()

    # 5. Inference Loop
    all_preds = []
    all_fnames = []

    print(f"Starting inference on {len(test_loader.dataset)} test files...")

    with torch.no_grad():
        for i, (images, _, fnames) in enumerate(test_loader):
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Get predictions (Argmax)
            # Softmax is monotonic, so argmax on logits is equivalent to argmax on probabilities
            _, preds = torch.max(outputs, dim=1)

            # Move to CPU and collect
            all_preds.extend(preds.cpu().numpy())
            all_fnames.extend(fnames)

    # 6. Post-processing
    # Map indices back to label strings
    predicted_labels = [IDX2LABEL[idx] for idx in all_preds]

    # Create Submission DataFrame
    df_submission = pd.DataFrame({"fname": all_fnames, "label": predicted_labels})

    # 7. Save Output
    save_path = Config.submission_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_submission.to_csv(save_path, index=False)

    print(f"Inference complete. Submission saved to: {save_path}")
    print("Submission Preview:")
    print(df_submission.head())
