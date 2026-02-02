import os
import torch
from library.config import Config
from library.model import InkUNet
from library.utils import generate_submission


def predict_and_submit(checkpoint_path=None, threshold=0.5):
    """
    Loads the trained model and generates the submission file for the test set.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to Config.CHECKPOINT_DIR/best_model.pth.
        threshold (float, optional): Optimal threshold for binarization. Defaults to 0.5.
    """
    # 1. Setup Environment
    Config.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Initialize Model
    # Instantiate the U-Net with the z-dimension defined in Config
    model = InkUNet(z_dim=Config.Z_DIM).to(device)

    # 3. Load Model Weights
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(checkpoint_path):
        # Load state dict with map_location to handle device discrepancies
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint file not found at {checkpoint_path}. Using random initialization."
        )

    # 4. Generate Submission
    # The generate_submission utility handles:
    # - Reading the test metadata (Config.TEST_METADATA_PATH)
    # - Loading 3D volume patches and batching them
    # - Stitching predicted probability maps into full fragment images
    # - Applying the binary threshold
    # - RLE encoding the binary mask
    # - Saving the result to Config.SUBMISSION_PATH
    generate_submission(model, device, threshold=threshold)
