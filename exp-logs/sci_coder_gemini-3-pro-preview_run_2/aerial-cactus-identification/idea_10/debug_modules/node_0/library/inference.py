import os
import torch
import numpy as np
from library.config import SEEDS, BATCH_SIZE, MODEL_DIR, NUM_WORKERS
from library.utils import get_device, save_submission
from library.dataset import get_dataloaders
from library.model import CustomNarrowSEMultiScaleResNet, predict_with_tta


def run_inference(seeds=SEEDS, batch_size=BATCH_SIZE, load_cached_data=True):
    """
    Runs the inference pipeline using the trained models.

    Args:
        seeds (list): List of seeds to identify model checkpoints.
        batch_size (int): Batch size for the test loader.
        load_cached_data (bool): Whether to use cached data for the dataloader.
    """
    device = get_device()

    # 1. Get Test DataLoader
    # We only need the test_loader and test_ids.
    # get_dataloaders returns: train_loader, val_loader, test_loader, test_ids
    _, _, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # 2. Load Models
    models = []
    print(f"Loading models for seeds: {seeds}")

    for seed in seeds:
        model_path = os.path.join(MODEL_DIR, f"model_seed_{seed}.pth")
        if os.path.exists(model_path):
            try:
                model = CustomNarrowSEMultiScaleResNet().to(device)
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                models.append(model)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model for seed {seed}: {e}")
        else:
            print(f"Warning: Model checkpoint not found at {model_path}. Skipping.")

    if not models:
        print("No valid models found. Aborting inference.")
        return

    print(
        f"Starting inference with {len(models)} models using Test Time Augmentation (TTA)..."
    )

    final_predictions = []

    # 3. Inference Loop
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)

            # Tensor to accumulate probabilities from all models
            # Shape: (Batch_Size, 1)
            batch_preds_sum = torch.zeros((images.size(0), 1), device=device)

            for model in models:
                # predict_with_tta returns probabilities (sigmoid applied)
                # It handles Original, H-Flip, and V-Flip internally
                preds = predict_with_tta(model, images, device)
                batch_preds_sum += preds

            # Average across the ensemble
            batch_preds_avg = batch_preds_sum / len(models)

            # Flatten and convert to numpy
            final_predictions.extend(batch_preds_avg.cpu().numpy().flatten())

    # 4. Save Submission
    print(f"Generated {len(final_predictions)} predictions.")
    save_submission(test_ids, final_predictions, "submission.csv")
    print("Inference complete. Submission saved to submission.csv")
