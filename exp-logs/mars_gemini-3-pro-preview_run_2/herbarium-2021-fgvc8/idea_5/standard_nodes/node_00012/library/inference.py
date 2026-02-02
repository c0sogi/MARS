import os
import torch
from library.config import Config
from library.utils import seed_everything, compute_class_priors, generate_submission
from library.dataset import get_dataloaders
from library.model import HierarchicalConvNeXt


def run_inference(debug_size=None):
    """
    Orchestrates the inference pipeline:
    1. Sets up the environment (seed, device).
    2. Loads the best model checkpoint.
    3. Prepares the test dataloader.
    4. Computes class priors for Post-Hoc Logit Adjustment.
    5. Generates the submission file using the utility function.

    Args:
        debug_size (int, optional): If set, limits the test set size for debugging.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Initialize Model architecture
    # We set pretrained=False here because we are about to load our own trained weights
    # avoiding unnecessary downloads of ImageNet weights.
    model = HierarchicalConvNeXt(pretrained=False)
    model.to(device)

    # 3. Load Best Model Weights
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.BEST_MODEL_PATH}"
        )

    print(f"Loading model weights from {Config.BEST_MODEL_PATH}...")
    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 4. Prepare Data Loaders
    # We pass stage=1, but it doesn't affect the test_loader construction.
    print(f"Preparing test dataloader (debug_size={debug_size})...")
    _, _, test_loader = get_dataloaders(stage=1, debug_size=debug_size)

    # 5. Compute Class Priors
    # Required for Post-Hoc Logit Adjustment to handle class imbalance
    print("Loading class priors for logit adjustment...")
    class_priors = compute_class_priors(load_cached_data=True)

    # 6. Generate Submission
    # The utility function handles the inference loop, logit adjustment, and CSV saving
    print("Generating predictions...")
    generate_submission(
        model=model,
        data_loader=test_loader,
        device=device,
        class_priors=class_priors,
        output_path=Config.SUBMISSION_PATH,
    )

    print(
        f"Inference completed successfully. Submission saved to {Config.SUBMISSION_PATH}"
    )
