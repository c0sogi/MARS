import os
import torch
import pandas as pd
import numpy as np

from library.config import Config
from library.dataset import get_test_loader
from library.models import get_model


def predict_and_submit(model=None):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model (torch.nn.Module, optional): A trained model instance. If None,
                                           loads the model from disk.
    """
    device = Config.DEVICE

    # 1. Load Test Metadata to get image_ids
    # The loader is deterministic (shuffle=False), so order matches metadata
    df_test = pd.read_csv(Config.TEST_METADATA)
    image_ids = df_test["image_id"].values

    print(f"Starting Inference on {len(image_ids)} test images...")

    # 2. Prepare Model
    if model is None:
        # Initialize architecture without downloading ImageNet weights (pretrained=False)
        # since we are loading our own trained weights immediately.
        model = get_model(pretrained=False)

        # Load trained weights
        model_path = os.path.join(Config.OUTPUT_DIR, "final_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_path}")

    model.eval()
    model.to(device)

    # 3. Prepare Data Loader
    test_loader = get_test_loader()

    # 4. Inference Loop
    all_probs = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)

            # Apply Softmax to get probabilities
            probs = torch.softmax(logits, dim=1)

            # Move to CPU and store
            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    # Shape: (N_samples, N_classes)
    final_probs = np.vstack(all_probs)

    # 5. Format Submission
    # Create DataFrame with image_id
    submission_df = pd.DataFrame({"image_id": image_ids})

    # Add probability columns
    # Config.TARGET_COLS order matches the model output order
    for idx, col_name in enumerate(Config.TARGET_COLS):
        submission_df[col_name] = final_probs[:, idx]

    # 6. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
