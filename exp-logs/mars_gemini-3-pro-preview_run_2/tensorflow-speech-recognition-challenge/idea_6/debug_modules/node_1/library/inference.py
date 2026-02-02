import os
import torch
import pandas as pd
import numpy as np
from library import config
from library import utils
from library import dataset
from library import model as model_lib


def generate_submission(
    model_path=config.MODEL_CHECKPOINT_PATH,
    output_path=config.SUBMISSION_PATH,
    batch_size=config.BATCH_SIZE,
    device=config.DEVICE,
):
    """
    Generates predictions for the test dataset using the trained model and saves them to a CSV file.

    Args:
        model_path (str): Path to the saved model checkpoint (.pth file).
        output_path (str): Path where the submission CSV should be saved.
        batch_size (int): Number of samples per batch during inference.
        device (str): Computation device ('cpu' or 'cuda').

    Returns:
        pd.DataFrame: The dataframe containing the generated predictions.
    """
    # 1. Setup
    utils.set_seed(config.SEED)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"Initializing inference on device: {device}")

    # 2. Load Model
    # Initialize the architecture
    model = model_lib.ConvNeXtSpeech(
        num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED
    )

    # Load the best weights
    model, best_val_acc = utils.load_checkpoint(
        model, filename=model_path, device=device
    )
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"Model Best Validation Accuracy: {best_val_acc}")

    # 3. Load Test Data
    # We use the library function to ensure preprocessing matches training
    test_loader = dataset.get_test_dataloader(
        test_csv=config.TEST_METADATA_PATH,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
    )

    print(f"Test DataLoader ready. Total samples: {len(test_loader.dataset)}")

    # 4. Inference Loop
    all_fnames = []
    all_preds = []

    print("Starting inference...")

    with torch.no_grad():
        for specs, _, fnames in test_loader:
            specs = specs.to(device)

            # Forward pass: (Batch, 1, Freq, Time) -> (Batch, Num_Classes)
            outputs = model(specs)

            # Get predicted class indices
            # The model is trained on 12 classes including 'unknown' and 'silence',
            # so we take the direct argmax.
            _, preds = torch.max(outputs, 1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_fnames.extend(fnames)

    # 5. Generate Submission File
    # Map indices back to string labels
    pred_labels = [config.ID2LABEL[idx] for idx in all_preds]

    df_submission = pd.DataFrame({"fname": all_fnames, "label": pred_labels})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df_submission.to_csv(output_path, index=False)

    print(f"Inference complete.")
    print(f"Submission saved to {output_path} with {len(df_submission)} rows.")

    return df_submission
