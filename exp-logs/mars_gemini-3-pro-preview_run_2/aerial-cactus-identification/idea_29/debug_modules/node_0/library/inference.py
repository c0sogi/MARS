import os
import numpy as np
import pandas as pd
import torch
from library.model import WideAntiAliasedRes2NeXt
from library.dataset import get_dataloaders, get_test_ids
from library.utils import seed_everything


def predict_tta(model, loader, device):
    """
    Generates predictions using Test Time Augmentation (TTA).
    Averages predictions from:
    1. Original image
    2. Horizontally flipped image
    3. Vertically flipped image

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        np.array: Flattened array of predicted probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)

            # 1. Original
            out = torch.sigmoid(model(images))

            # 2. Horizontal Flip (dim 3 is width for NCHW)
            out_h = torch.sigmoid(model(torch.flip(images, [3])))

            # 3. Vertical Flip (dim 2 is height for NCHW)
            out_v = torch.sigmoid(model(torch.flip(images, [2])))

            # Average probabilities
            p = (out + out_h + out_v) / 3.0
            preds.extend(p.cpu().numpy().flatten())

    return np.array(preds)


def generate_ensemble_predictions(
    seeds=[0, 1, 2, 3, 4],
    work_dir="./working/idea_29",
    submission_path="./submission/submission.csv",
    batch_size=64,
    num_workers=2,
    load_cached_data=True,
):
    """
    Loads multiple trained model checkpoints, generates TTA predictions for each,
    averages them (ensemble), and saves the final submission file.

    Args:
        seeds (list): List of seed integers corresponding to the trained models.
        work_dir (str): Directory containing the saved model checkpoints.
        submission_path (str): Path to save the generated submission CSV.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker subprocesses for data loading.
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Load Test Data
    # get_dataloaders returns (train, val, test). We only need test.
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=load_cached_data,
    )

    # Get Test IDs to match predictions
    test_ids = get_test_ids(load_cached_data=load_cached_data)

    # Initialize accumulator for ensemble predictions
    final_preds = np.zeros(len(test_ids))
    models_found = 0

    print(f"Starting inference on device: {device}")
    print(f"Ensembling models from seeds: {seeds}")

    for seed in seeds:
        model_path = os.path.join(work_dir, f"model_seed_{seed}.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model checkpoint not found for seed {seed} at {model_path}. Skipping."
            )
            continue

        print(f"Processing Seed {seed}...")

        # Ensure reproducibility for any stochastic operations (though inference is deterministic)
        seed_everything(seed)

        # Initialize Model
        model = WideAntiAliasedRes2NeXt()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        # Generate Predictions with TTA
        seed_preds = predict_tta(model, test_loader, device)

        # Accumulate
        final_preds += seed_preds
        models_found += 1

    if models_found == 0:
        raise RuntimeError(
            "No model checkpoints were found. Cannot generate submission."
        )

    # Average predictions
    final_preds /= models_found

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})

    # Save to CSV
    df_sub.to_csv(submission_path, index=False)
    print(f"Ensemble prediction complete. Submission saved to {submission_path}")
