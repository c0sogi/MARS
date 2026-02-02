import os
import torch
import pandas as pd
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import RetinopathyModel, generate_submission


def inference_fn(
    model_path="./working/idea_3/best_model.pth",
    output_path="./submission/submission.csv",
    batch_size=8,
    image_size=768,
    device=None,
):
    """
    Runs inference on the test set using a trained model and generates a submission file.

    Args:
        model_path (str): Path to the trained model weights (.pth file).
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        image_size (int): Input image size (should match the resolution used in training).
        device (torch.device, optional): Device to run inference on. If None, automatically detects GPU/CPU.
    """
    # 1. Setup Environment
    seed_everything(42)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Inference Device: {device}")

    # 2. Load Data
    # We use get_dataloaders to leverage the existing caching and preprocessing pipeline.
    # We only need the test_loader for inference. The function handles loading metadata from ./metadata.
    print(f"Loading test data (Image Size: {image_size})...")
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size,
        image_size=image_size,
        load_cached_data=True,
        input_dir="./input",
        metadata_dir="./metadata",
    )

    # 3. Initialize Model
    # pretrained=False is used because we are about to load our own fine-tuned weights.
    # Initializing with ImageNet weights would be redundant.
    print("Initializing model...")
    model = RetinopathyModel(pretrained=False)
    model.to(device)

    # 4. Load Weights
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # If the specific model path doesn't exist, we cannot proceed with valid inference.
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Cannot proceed with inference."
        )

    # 5. Generate Submission
    # The generate_submission function from library.model handles:
    # - Iterating through the test_loader
    # - Predicting continuous scores
    # - Clipping scores to [0, 4] and rounding to the nearest integer
    # - Mapping predictions to id_codes from ./metadata/test.csv
    # - Saving the result to output_path
    print("Generating submission...")
    generate_submission(model, test_loader, device, output_path=output_path)

    print("Inference complete.")
