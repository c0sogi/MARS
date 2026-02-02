import os
import torch
from library.config import Config
from library.utils import set_seed, load_checkpoint, create_submission
from library.model import SRDN
from library.data_loader import get_test_dataloader


def predict_full_image(model, img_tensor, device):
    """
    Takes a full-resolution noisy image and the trained model, performs a forward pass
    to predict the noise, and subtracts it from the input to recover the clean image.
    Note: The SRDN model's forward method encapsulates the subtraction logic.

    Args:
        model (torch.nn.Module): The trained SRDN model.
        img_tensor (torch.Tensor): The noisy input image tensor of shape (1, 1, H, W).
        device (str): The device to run inference on.

    Returns:
        torch.Tensor: The predicted clean image tensor.
    """
    model.eval()
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        # The model returns the clean image directly: Clean = Input - Predicted_Noise
        clean_pred = model(img_tensor)

    return clean_pred


def generate_submission_file(model_path=None, output_path=None, device=None):
    """
    Generates predictions for the test set and saves the submission CSV.

    Args:
        model_path (str, optional): Path to the model checkpoint. Defaults to Config.WORKING_DIR/best_model.pth.
        output_path (str, optional): Path to save the submission CSV. Defaults to Config.SUBMISSION_PATH.
        device (str, optional): Device to run inference on. Defaults to Config.DEVICE.
    """
    # Set defaults if not provided
    if model_path is None:
        model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if output_path is None:
        output_path = Config.SUBMISSION_PATH
    if device is None:
        device = Config.DEVICE

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"Initializing inference on {device}...")

    # Initialize Model
    model = SRDN().to(device)

    # Load Weights
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        load_checkpoint(model_path, model, device=device)
    else:
        print(
            f"Warning: Model checkpoint not found at {model_path}. "
            "Predictions will be based on random initialization (for debugging only)."
        )

    # Prepare Test Data
    print("Loading test data...")
    test_loader = get_test_dataloader()

    predictions = {}

    print(f"Processing {len(test_loader)} test images...")

    # Inference Loop
    for i, (img, img_id_tuple) in enumerate(test_loader):
        # img_id_tuple is a tuple because the loader has batch_size=1
        img_id = img_id_tuple[0]

        # Predict clean image
        clean_tensor = predict_full_image(model, img, device)

        # Remove batch dimension: (1, 1, H, W) -> (1, H, W)
        # create_submission expects (C, H, W) or (H, W)
        predictions[img_id] = clean_tensor.squeeze(0)

    # Generate Submission File
    print(f"Generating submission file at {output_path}...")
    create_submission(predictions, output_path)
    print("Submission generation complete.")
