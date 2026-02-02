import os
import torch
import numpy as np
import pandas as pd
from library import config, utils, data_loader, model


def load_ensemble(device):
    """
    Loads the 5-fold ensemble of AC-WIV models.
    Returns a list of models in evaluation mode.
    """
    models = []
    for fold_idx in range(config.NUM_FOLDS):
        # Initialize model architecture
        net = model.ACWIVNet(
            backbone_name=config.BACKBONE,
            pretrained=False,  # No need to download weights, we load checkpoint
            input_channels=config.INPUT_CHANNELS,
        )
        net = net.to(device)

        # Construct checkpoint path
        checkpoint_name = f"best_model_fold{fold_idx}.pth"
        checkpoint_path = os.path.join(config.WORKING_DIR, checkpoint_name)

        # Load weights
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint for Fold {fold_idx}: {checkpoint_path}")
            utils.load_checkpoint(net, checkpoint_path, device=device)
            net.eval()
            models.append(net)
        else:
            print(
                f"Warning: Checkpoint for Fold {fold_idx} not found at {checkpoint_path}. Skipping."
            )

    if not models:
        raise RuntimeError("No checkpoints found. Cannot perform inference.")

    return models


def predict(load_cached_data=True):
    """
    Main inference function.
    1. Loads test data (cached or processed from scratch).
    2. Loads ensemble models.
    3. Generates predictions.
    4. Saves submission.csv.
    """
    print("Starting Inference...")

    # 1. Prepare Data
    # get_test_dataloader handles caching of processed numpy arrays internally
    test_loader, test_ids = data_loader.get_test_dataloader(
        load_cached_data=load_cached_data
    )

    # 2. Load Models
    device = config.DEVICE
    models = load_ensemble(device)
    print(f"Loaded {len(models)} models for ensemble inference.")

    # 3. Inference Loop
    all_probs = []

    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            images = images.to(device)

            # Ensemble Prediction
            batch_probs = []
            for net in models:
                logits = net(images)
                probs = torch.sigmoid(logits)
                batch_probs.append(probs.cpu().numpy())

            # Average across folds (Shape: [n_models, batch_size, 1])
            batch_probs = np.array(batch_probs)
            avg_probs = np.mean(batch_probs, axis=0)  # Shape: [batch_size, 1]

            all_probs.extend(avg_probs.flatten())

    # 4. Create Submission
    # Ensure lengths match
    if len(all_probs) != len(test_ids):
        print(
            f"Warning: Number of predictions ({len(all_probs)}) does not match number of IDs ({len(test_ids)})."
        )

    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": all_probs})

    # Sort by ID just in case, though order is preserved
    submission_df = submission_df.sort_values("BraTS21ID")

    # Save
    save_path = config.SUBMISSION_PATH
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(submission_df.head())
