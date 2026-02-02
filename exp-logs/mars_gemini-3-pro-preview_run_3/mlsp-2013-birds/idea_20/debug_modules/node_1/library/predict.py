import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import load_data, BirdDataset, get_transforms
from library.models import get_model
from library.utils import seed_everything


def predict_with_tta(model, device, loaders):
    """
    Generates predictions for a single model using TTA (Original, Left, Right).
    Averages the probabilities across the 3 views.

    Args:
        model: Loaded PyTorch model.
        device: Torch device.
        loaders: List of 3 DataLoaders [Original, Left, Right].

    Returns:
        np.ndarray: Averaged predictions (N, Num_Classes).
    """
    model.eval()

    # Store predictions for each view
    tta_preds = []

    with torch.no_grad():
        for loader in loaders:
            view_preds = []
            for images in loader:
                images = images.to(device)
                outputs = model(images)
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                view_preds.append(probs.cpu().numpy())

            # Concatenate batches for this view
            if len(view_preds) > 0:
                tta_preds.append(np.concatenate(view_preds, axis=0))
            else:
                # Handle empty loader case if necessary
                tta_preds.append(np.empty((0, Config.NUM_CLASSES)))

    # Stack and average across the 3 views
    # shape: (3, N, Num_Classes)
    stacked_preds = np.array(tta_preds)
    avg_preds = np.mean(stacked_preds, axis=0)

    return avg_preds


def run_inference():
    """
    Main inference routine.
    Loads data, prepares TTA, runs ensemble inference, and generates submission.
    """
    seed_everything(Config.SEED)
    Config.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")

    # 1. Load Test Data
    # images shape: (N, 224, 224, 3), dtype: uint8
    # We ignore the labels returned by load_data for the test set
    print("Loading test data...")
    images, _ = load_data("test")

    if len(images) == 0:
        print("Error: No test images loaded.")
        return

    # 2. Prepare TTA Images (Deterministic Shifts)
    print("Preparing TTA views...")
    h, w, c = images.shape[1:]
    shift_pixels = int(w * Config.SHIFT_LIMIT)

    # View 1: Original
    imgs_orig = images

    # View 2: Left Shift (Content moves left, pad right)
    imgs_left = np.zeros_like(images)
    # Copy content from [shift:] to [:-shift]
    imgs_left[:, :, :-shift_pixels, :] = images[:, :, shift_pixels:, :]

    # View 3: Right Shift (Content moves right, pad left)
    imgs_right = np.zeros_like(images)
    # Copy content from [:-shift] to [shift:]
    imgs_right[:, :, shift_pixels:, :] = images[:, :, :-shift_pixels, :]

    # 3. Create DataLoaders
    # We use the 'val' transform which performs Resize (redundant but safe) + Normalize + ToTensor
    transform = get_transforms("val")

    # Create Datasets
    # We pass labels=None so the dataset returns only images
    ds_orig = BirdDataset(imgs_orig, labels=None, transform=transform)
    ds_left = BirdDataset(imgs_left, labels=None, transform=transform)
    ds_right = BirdDataset(imgs_right, labels=None, transform=transform)

    # Create DataLoaders
    # We can use a larger batch size for inference
    inf_batch_size = Config.BATCH_SIZE * 2

    loaders = [
        DataLoader(
            ds_orig,
            batch_size=inf_batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        ),
        DataLoader(
            ds_left,
            batch_size=inf_batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        ),
        DataLoader(
            ds_right,
            batch_size=inf_batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        ),
    ]

    # 4. Ensemble Inference
    ensemble_preds = np.zeros((len(images), Config.NUM_CLASSES))
    model_count = 0

    print("Starting Ensemble Inference...")

    for model_name in Config.MODEL_ARCHITECTURES:
        for fold in range(Config.NUM_FOLDS):
            ckpt_filename = f"{model_name}_fold_{fold}_best.pth"
            ckpt_path = os.path.join(Config.CHECKPOINT_DIR, ckpt_filename)

            if not os.path.exists(ckpt_path):
                print(f"Warning: Checkpoint not found: {ckpt_path}. Skipping.")
                continue

            # Load Model
            # pretrained=False because we are loading a full state_dict of a trained model
            try:
                model = get_model(model_name, pretrained=False)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)

                # Predict
                preds = predict_with_tta(model, device, loaders)
                ensemble_preds += preds
                model_count += 1

                print(f"Processed {model_name} (Fold {fold})")

                # Cleanup to save memory
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {model_name} fold {fold}: {e}")

    if model_count == 0:
        raise RuntimeError("No models were successfully loaded and executed.")

    # Average predictions
    final_preds = ensemble_preds / model_count
    print(f"Inference complete. Averaged over {model_count} models.")

    # 5. Generate Submission File
    print("Generating submission file...")

    # Load test metadata to get the correct rec_id for each image
    # The order of load_data("test") matches the order of rows in metadata/test.csv
    test_df = pd.read_csv(Config.TEST_METADATA)
    rec_ids = test_df["rec_id"].values

    if len(rec_ids) != len(final_preds):
        raise ValueError(
            f"Mismatch: {len(rec_ids)} rec_ids vs {len(final_preds)} predictions."
        )

    submission_rows = []

    for i, rec_id in enumerate(rec_ids):
        # Get probabilities for this recording (19 classes)
        probs = final_preds[i]

        for species_idx, prob in enumerate(probs):
            # Construct Id: rec_id * 100 + species_number
            row_id = int(rec_id * 100 + species_idx)
            submission_rows.append({"Id": row_id, "Probability": prob})

    submission_df = pd.DataFrame(submission_rows)

    # Ensure sorted by Id (good practice)
    submission_df = submission_df.sort_values("Id")

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
