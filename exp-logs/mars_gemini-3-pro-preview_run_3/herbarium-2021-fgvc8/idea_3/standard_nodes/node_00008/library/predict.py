import os
import torch
import pandas as pd
from torch.optim.swa_utils import update_bn
from library.config import Config
from library.dataset import get_dataloaders
from library.model import HerbariumConvNeXt, generate_submission


def run_prediction(debug=False, load_cached_data=True):
    """
    Manages the inference process for the Plant Species Classification task.

    This function:
    1. Sets up the configuration (debug mode).
    2. Loads metadata and initializes DataLoaders.
    3. Loads the trained model (prioritizing SWA model).
    4. Updates BatchNorm statistics if using SWA.
    5. Generates predictions on the test set and saves to CSV.

    Args:
        debug (bool): If True, runs on a subset of data for debugging.
        load_cached_data (bool): Whether to use cached sampler weights for the dataloader.
    """
    # 1. Update Configuration
    if debug:
        Config.DEBUG = True
        print(f"Debug mode enabled. Using {Config.DEBUG_SAMPLE_SIZE} samples.")

    device = torch.device(Config.DEVICE)

    # 2. Load Metadata
    print("Loading metadata...")
    if (
        not os.path.exists(Config.TRAIN_CSV)
        or not os.path.exists(Config.VAL_CSV)
        or not os.path.exists(Config.TEST_CSV)
    ):
        raise FileNotFoundError(
            "Metadata files not found. Please ensure metadata generation script has been run."
        )

    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 3. Initialize DataLoaders
    # train_loader is needed for SWA BatchNorm update
    # test_loader is needed for inference
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # 4. Initialize Model Architecture
    # We set pretrained=False because we are about to load our own trained weights
    print(f"Initializing model architecture: {Config.MODEL_NAME}")
    model = HerbariumConvNeXt(pretrained=False)
    model = model.to(device)

    # 5. Load Weights
    swa_path = Config.SWA_MODEL_SAVE_PATH
    best_path = Config.MODEL_SAVE_PATH

    if os.path.exists(swa_path):
        print(f"Found SWA model checkpoint at {swa_path}. Loading...")
        state_dict = torch.load(swa_path, map_location=device)

        # Clean state_dict if it contains SWA-specific keys like 'n_averaged'
        # which are not part of the base model architecture
        if "n_averaged" in state_dict:
            del state_dict["n_averaged"]

        # Remove 'module.' prefix added by AveragedModel wrapper
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

        model.load_state_dict(state_dict, strict=True)

        # Update BatchNorm statistics
        # This pass is necessary to ensure the SWA model's batch norm layers
        # reflect the statistics of the training data.
        print("Updating BatchNorm statistics (this may take a while)...")
        model.train()  # update_bn requires model to be in train mode
        update_bn(train_loader, model, device=device)
        print("BatchNorm statistics updated.")

    elif os.path.exists(best_path):
        print(f"Found best model checkpoint at {best_path}. Loading...")
        state_dict = torch.load(best_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)

    else:
        raise FileNotFoundError(
            f"No trained model found. Checked {swa_path} and {best_path}."
        )

    # 6. Generate Submission
    # This function handles the evaluation loop and saving to CSV
    generate_submission(model, test_loader)

    print("Inference process completed successfully.")
