import os
import torch
import numpy as np
from library import config, utils, model, dataset


def predict_single_model(model_instance, test_loader, device):
    """
    Generates predictions for a single model using D4 Test-Time Augmentation (TTA).
    Returns a dictionary mapping image IDs to predicted numpy arrays.

    Args:
        model_instance (nn.Module): The trained neural network model.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Computation device ('cpu' or 'cuda').

    Returns:
        dict: {img_id (str): prediction (np.ndarray)}
    """
    model_instance.eval()
    predictions = {}

    with torch.no_grad():
        for data in test_loader:
            # dataset.py in test mode returns (noisy_tensor, metadata_dict)
            if len(data) == 2:
                noisy, meta = data
            else:
                continue

            # Extract metadata
            # meta['id'] is a list (batch size 1), so we take the first element
            img_id = str(meta["id"][0])
            orig_h = meta["orig_h"].item()
            orig_w = meta["orig_w"].item()

            x = noisy.to(device)  # Shape: (1, 1, H, W)

            # --- D4 Group TTA (8 views) ---

            # Group 1: No Transpose (H, W) -> [Original, Rot180, Flip, Flip+Rot180]
            # Rotations are counter-clockwise. k=2 is 180 degrees.
            x_rot180 = torch.rot90(x, 2, [2, 3])
            x_flip = torch.flip(x, [3])  # Horizontal flip
            x_flip_rot180 = torch.rot90(x_flip, 2, [2, 3])

            batch1 = torch.cat([x, x_rot180, x_flip, x_flip_rot180], dim=0)
            out1 = model_instance(batch1)

            # Inverse transformations for Group 1
            y1 = out1[0:1]
            y2 = torch.rot90(out1[1:2], 2, [2, 3])  # Inverse of Rot180 is Rot180
            y3 = torch.flip(out1[2:3], [3])  # Inverse of Flip is Flip
            # Inverse of (Flip->Rot180) is (Rot180->Flip)
            y4 = torch.flip(torch.rot90(out1[3:4], 2, [2, 3]), [3])

            # Group 2: Transpose (W, H) -> [Rot90, Rot270, Flip+Rot90, Flip+Rot270]
            x_rot90 = torch.rot90(x, 1, [2, 3])
            x_rot270 = torch.rot90(x, 3, [2, 3])
            x_flip_rot90 = torch.rot90(x_flip, 1, [2, 3])
            x_flip_rot270 = torch.rot90(x_flip, 3, [2, 3])

            batch2 = torch.cat([x_rot90, x_rot270, x_flip_rot90, x_flip_rot270], dim=0)
            out2 = model_instance(batch2)

            # Inverse transformations for Group 2
            # Inverse of Rot90 is Rot270 (k=3)
            y5 = torch.rot90(out2[0:1], 3, [2, 3])
            # Inverse of Rot270 is Rot90 (k=1)
            y6 = torch.rot90(out2[1:2], 1, [2, 3])
            # Inverse of (Flip->Rot90) is (Rot270->Flip) -> Flip(Rot270)
            y7 = torch.flip(torch.rot90(out2[2:3], 3, [2, 3]), [3])
            # Inverse of (Flip->Rot270) is (Rot90->Flip) -> Flip(Rot90)
            y8 = torch.flip(torch.rot90(out2[3:4], 1, [2, 3]), [3])

            # Average all 8 views
            y_avg = (y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8) / 8.0

            # Crop back to original dimensions (remove padding)
            pred_img = y_avg[0, 0, :orig_h, :orig_w].cpu().numpy()
            predictions[img_id] = pred_img

    return predictions


def run_ensemble_inference(seeds=config.SEEDS, output_path=config.SUBMISSION_FILE_PATH):
    """
    Runs inference using an ensemble of models trained with different seeds.
    Aggregates predictions by averaging and saves the submission file.

    Args:
        seeds (list): List of random seeds corresponding to trained models.
        output_path (str): Path to save the final submission CSV.
    """
    # Ensure deterministic behavior
    utils.set_seed(42)
    device = config.DEVICE

    # Load Test Data
    # We use the factory function but only need the test_loader
    # load_cached=True ensures we use the cache mechanism in dataset.py
    _, _, test_loader = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, load_cached=True
    )

    ensemble_preds = {}
    valid_seeds = []

    print(f"Starting ensemble inference with {len(seeds)} models...")

    for seed in seeds:
        checkpoint_path = config.get_checkpoint_path(seed)

        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Checkpoint for seed {seed} not found at {checkpoint_path}. Skipping."
            )
            continue

        print(f"Processing model with seed {seed}...")

        # Initialize the model architecture
        net = model.WideBottleneckUNet(
            n_channels=config.IN_CHANNELS, n_classes=config.OUT_CHANNELS
        )

        # Load model weights
        state_dict = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(state_dict)
        net.to(device)

        # Generate predictions for this specific model
        preds = predict_single_model(net, test_loader, device)

        # Accumulate predictions
        if not ensemble_preds:
            ensemble_preds = preds
        else:
            for img_id, pred_arr in preds.items():
                ensemble_preds[img_id] += pred_arr

        valid_seeds.append(seed)

        # Clean up to save memory
        del net
        del state_dict
        torch.cuda.empty_cache()

    if not valid_seeds:
        print("Error: No valid models found. Cannot generate submission.")
        return

    # Average the accumulated predictions
    num_models = len(valid_seeds)
    print(f"Averaging predictions from {num_models} models...")

    for img_id in ensemble_preds:
        ensemble_preds[img_id] /= float(num_models)

        # Explicitly clip to [0, 1] range to ensure valid pixel intensities
        ensemble_preds[img_id] = np.clip(ensemble_preds[img_id], 0, 1)

    # Generate and save the submission file
    print(f"Saving submission to {output_path}...")
    utils.create_submission(ensemble_preds, output_path)
    print("Inference complete.")
