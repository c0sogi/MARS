import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import OrderedDict

from library.config import (
    MODEL_EFFICIENTNET,
    MODEL_CONVNEXT,
    CLASS_LABELS,
    SUBMISSION_PATH,
    NUM_WORKERS,
    BATCH_SIZE,
    SEED,
)
from library.utils import seed_everything
from library.data import get_dataloaders
from library.models import get_model


def load_model_weights(model, weights_path, device):
    """
    Loads weights into the model, handling potential 'module.' prefixes
    from SWA or DataParallel wrappers.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)

    # Check if the state_dict keys have 'module.' prefix (common with SWA/DataParallel)
    # and the model does not.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    # Load weights
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Original + HFlip + VFlip).

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Test data loader.
        device (torch.device): Computation device.

    Returns:
        tuple: (image_ids, probabilities_array)
    """
    all_ids = []
    all_probs = []

    # TTA: Horizontal Flip (dim 3) and Vertical Flip (dim 2)
    # Assuming input is (B, C, H, W)

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # 1. Original
            logits_orig = model(images)
            probs_orig = F.softmax(logits_orig, dim=1)

            # 2. Horizontal Flip
            images_h = torch.flip(images, dims=[3])
            logits_h = model(images_h)
            probs_h = F.softmax(logits_h, dim=1)

            # 3. Vertical Flip
            images_v = torch.flip(images, dims=[2])
            logits_v = model(images_v)
            probs_v = F.softmax(logits_v, dim=1)

            # Average probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            all_probs.append(avg_probs.cpu().numpy())
            all_ids.extend(ids)

    return all_ids, np.concatenate(all_probs, axis=0)


def ensemble_predictions(effnet_weights_path, convnext_weights_path, debug=False):
    """
    Generates predictions for the test set using an ensemble of EfficientNet and ConvNeXt.
    Saves the result to submission.csv.

    Args:
        effnet_weights_path (str): Path to EfficientNet weights.
        convnext_weights_path (str): Path to ConvNeXt weights.
        debug (bool): If True, runs on a subset of data.
    """
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}. Debug={debug}")

    # 1. Get Data
    # We only need the test loader here
    _, _, test_loader = get_dataloaders(
        debug=debug, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # 2. EfficientNet Prediction
    print(f"Loading EfficientNet from {effnet_weights_path}...")
    # Initialize with pretrained=False for speed since we are loading custom weights
    model_eff = get_model(MODEL_EFFICIENTNET, pretrained=False)
    model_eff = load_model_weights(model_eff, effnet_weights_path, device)

    print("Generating EfficientNet predictions with TTA...")
    ids, probs_eff = predict_with_tta(model_eff, test_loader, device)

    # Free memory
    del model_eff
    torch.cuda.empty_cache()

    # 3. ConvNeXt Prediction
    print(f"Loading ConvNeXt from {convnext_weights_path}...")
    model_conv = get_model(MODEL_CONVNEXT, pretrained=False)
    model_conv = load_model_weights(model_conv, convnext_weights_path, device)

    print("Generating ConvNeXt predictions with TTA...")
    # We assume IDs come out in the same order because DataLoader is sequential and deterministic
    _, probs_conv = predict_with_tta(model_conv, test_loader, device)

    del model_conv
    torch.cuda.empty_cache()

    # 4. Ensemble (Unweighted Average)
    print("Ensembling predictions...")
    final_probs = (probs_eff + probs_conv) / 2.0

    # 5. Create Submission DataFrame
    df_sub = pd.DataFrame({"image_id": ids})

    # Assign probabilities to class columns
    for i, label in enumerate(CLASS_LABELS):
        df_sub[label] = final_probs[:, i]

    # 6. Save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

    # Print first few rows for verification
    print("\nSubmission Head:")
    print(df_sub.head())
