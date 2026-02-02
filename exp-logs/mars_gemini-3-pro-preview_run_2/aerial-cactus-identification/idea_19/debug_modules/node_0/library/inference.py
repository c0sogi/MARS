import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library import dataset, model, utils


def predict_with_tta(model_instance, loader, device):
    """
    Generates predictions for a single model instance using Test Time Augmentation.
    TTA Strategy: Original + Horizontal Flip + Vertical Flip.
    Returns a dictionary mapping image_id -> probability.
    """
    model_instance.eval()
    preds_dict = {}

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # 1. Original
            logits_orig = model_instance(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip (dim 3 is width for NCHW)
            images_h = torch.flip(images, dims=[3])
            logits_h = model_instance(images_h)
            probs_h = torch.sigmoid(logits_h)

            # 3. Vertical Flip (dim 2 is height for NCHW)
            images_v = torch.flip(images, dims=[2])
            logits_v = model_instance(images_v)
            probs_v = torch.sigmoid(logits_v)

            # Average probabilities (Soft Voting)
            probs_avg = (probs_orig + probs_h + probs_v) / 3.0

            # Store results
            # Move to CPU and numpy
            batch_probs = probs_avg.cpu().numpy().flatten()

            for img_id, prob in zip(ids, batch_probs):
                preds_dict[img_id] = prob

    return preds_dict


def ensemble_predictions(debug_sample_size=None):
    """
    Loads all trained model seeds, performs TTA inference, averages predictions,
    and generates the final submission file.
    """
    # 1. Setup
    device = torch.device(Config.DEVICE)
    print(f"Starting Inference on device: {device}")

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Load Test Data
    # We rely on the library function to get the loader
    _, _, test_loader = dataset.get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug_sample_size=debug_sample_size,
    )

    # 3. Ensemble Loop
    # We will accumulate probabilities in a dictionary: {id: sum_of_probs}
    ensemble_accumulator = {}
    seeds = Config.SEEDS
    successful_seeds = 0

    print(f"Ensembling over {len(seeds)} seeds: {seeds}")

    for seed in seeds:
        model_path = Config.get_model_path(seed)

        if not os.path.exists(model_path):
            print(f"Checkpoint for seed {seed} not found at {model_path}. Skipping.")
            continue

        print(f"Processing Seed {seed}...")

        # Initialize and Load Model
        net = model.CustomWideResNet()
        net.to(device)

        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict)

        # Predict with TTA
        seed_preds = predict_with_tta(net, test_loader, device)

        # Accumulate
        for img_id, prob in seed_preds.items():
            if img_id not in ensemble_accumulator:
                ensemble_accumulator[img_id] = 0.0
            ensemble_accumulator[img_id] += prob

        successful_seeds += 1

    if successful_seeds == 0:
        raise RuntimeError("No models were found. Cannot generate submission.")

    # 4. Finalize and Save
    print("Generating submission file...")

    # We load the test metadata to ensure we output in the expected order
    # and cover all IDs.
    df_template = pd.read_csv(Config.TEST_METADATA)

    if debug_sample_size is not None:
        df_template = df_template.iloc[:debug_sample_size]

    # Map predictions
    final_probs = []
    for img_id in df_template["id"]:
        if img_id in ensemble_accumulator:
            # Average: Sum / Count
            avg_prob = ensemble_accumulator[img_id] / successful_seeds
            final_probs.append(avg_prob)
        else:
            # Fallback (should not happen if loaders are consistent)
            print(f"Warning: ID {img_id} missing from predictions. Defaulting to 0.5")
            final_probs.append(0.5)

    df_template["has_cactus"] = final_probs

    # Keep only required columns
    submission_df = df_template[["id", "has_cactus"]]

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print("First 5 rows:")
    print(submission_df.head())
