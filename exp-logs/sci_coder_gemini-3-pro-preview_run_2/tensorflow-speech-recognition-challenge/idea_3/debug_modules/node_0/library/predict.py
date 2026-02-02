import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.model import ConvNeXtAudio
from library.dataset import get_dataloaders


def generate_submission(batch_size=Config.BATCH_SIZE, device=Config.DEVICE):
    """
    Generates predictions for the test set using the best trained model
    and saves the results to a CSV file.

    Args:
        batch_size (int): The batch size to use for inference.
        device (str): The device to run the model on ('cpu' or 'cuda').
    """
    # 1. Set Seed for Reproducibility
    set_seed(Config.SEED)

    # 2. Prepare Data
    print("Initializing Test DataLoader...")
    # Retrieve only the test loader
    loaders = get_dataloaders(batch_size=batch_size, num_workers=Config.NUM_WORKERS)
    test_loader = loaders["test"]

    # 3. Initialize Model
    print("Initializing Model architecture...")
    # We use pretrained=False here because we are about to load our own trained weights.
    # This avoids downloading the ImageNet weights unnecessarily.
    model = ConvNeXtAudio(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(device)

    # 4. Load Best Checkpoint
    print(f"Loading best model checkpoint from {Config.BEST_MODEL_PATH}...")
    checkpoint_data = load_checkpoint(model, path=Config.BEST_MODEL_PATH, device=device)

    if checkpoint_data is None:
        print(
            f"Warning: No checkpoint found at {Config.BEST_MODEL_PATH}. Using random initialization."
        )
    else:
        print(
            f"Model loaded successfully. (Epoch: {checkpoint_data.get('epoch')}, Val Acc: {checkpoint_data.get('val_acc')})"
        )

    # 5. Inference
    print("Starting Inference on Test Set...")
    model.eval()

    all_preds = []
    all_fnames = []

    with torch.no_grad():
        for inputs, _, fnames in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get predicted class indices
            # shape: (batch_size, num_classes) -> (batch_size,)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_fnames.extend(fnames)

    # 6. Map Indices to Labels
    predicted_labels = [Config.ID2LABEL[idx] for idx in all_preds]

    # 7. Create Submission DataFrame
    df_submission = pd.DataFrame({"fname": all_fnames, "label": predicted_labels})

    # 8. Save to CSV
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions generated: {len(df_submission)}")
