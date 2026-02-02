import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

from library.config import Config
from library.data import get_data, BirdDataset
from library.models import get_model
from library.utils import seed_everything, get_device


def get_tta_transforms(img_size):
    """
    Returns a list of transforms for Test-Time Augmentation (TTA).
    Includes: Original, Left Shift, and Right Shift.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Base operations
    resize = transforms.Resize((img_size, img_size))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=mean, std=std)

    # Shift magnitude: 10% of image width
    shift_px = int(img_size * 0.1)

    # 1. Original View
    t_orig = transforms.Compose([resize, to_tensor, normalize])

    # 2. Left Shift (Translate content left, i.e., tx < 0)
    # fill=0 ensures zero-padding for the new area
    t_left = transforms.Compose(
        [
            resize,
            transforms.Lambda(
                lambda img: F.affine(
                    img, angle=0, translate=(-shift_px, 0), scale=1.0, shear=0, fill=0
                )
            ),
            to_tensor,
            normalize,
        ]
    )

    # 3. Right Shift (Translate content right, i.e., tx > 0)
    t_right = transforms.Compose(
        [
            resize,
            transforms.Lambda(
                lambda img: F.affine(
                    img, angle=0, translate=(shift_px, 0), scale=1.0, shear=0, fill=0
                )
            ),
            to_tensor,
            normalize,
        ]
    )

    return [t_orig, t_left, t_right]


def predict_and_submit(config: Config):
    """
    Performs inference using the heterogeneous ensemble with TTA and generates the submission file.

    The process involves:
    1. Loading test data and metadata.
    2. Creating DataLoaders for each TTA view (Original, Left, Right).
    3. Iterating through all saved checkpoints (Arch x Fold x Rank).
    4. Accumulating predictions and averaging them.
    5. Formatting and saving the submission CSV.
    """
    # 1. Setup
    seed_everything(config.SEED)
    device = get_device()

    # 2. Load Data
    # get_data handles caching. We unpack only the test set ((imgs, lbls)).
    print("Loading test data...")
    _, _, (test_imgs, _) = get_data(config, load_cached_data=True)

    # Load Test Metadata to get corresponding rec_ids
    # The order in test_imgs corresponds strictly to the rows in test.csv
    test_df = pd.read_csv(config.TEST_METADATA)
    rec_ids = test_df["rec_id"].values

    if len(rec_ids) != len(test_imgs):
        raise ValueError(
            f"Mismatch between metadata rows ({len(rec_ids)}) and loaded images ({len(test_imgs)})"
        )

    # 3. Prepare Inference Resources
    num_samples = len(test_imgs)
    num_classes = config.NUM_CLASSES

    # Accumulator for probabilities: [N_samples, N_classes]
    ensemble_probs = np.zeros((num_samples, num_classes), dtype=np.float64)
    model_count = 0

    # Prepare DataLoaders for TTA
    # We pre-create these to avoid overhead inside the model loop
    tta_transforms = get_tta_transforms(config.IMG_SIZE)
    tta_loaders = []

    for transform in tta_transforms:
        # Create dataset with specific TTA transform
        # Labels are None, so dataset returns dummy zeros
        dataset = BirdDataset(test_imgs, labels=None, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        tta_loaders.append(loader)

    # 4. Ensemble Loop
    # Iterate over Architectures -> Folds -> Snapshots
    print("Starting Ensemble Inference...")

    for arch in config.ARCHITECTURES:
        for fold in range(config.NUM_FOLDS):
            for rank in range(config.TOP_K_CHECKPOINTS):

                # Construct checkpoint path
                checkpoint_path = config.get_checkpoint_path(arch, fold, rank)

                # Skip if checkpoint doesn't exist (e.g., training interrupted)
                if not os.path.exists(checkpoint_path):
                    continue

                print(f"Processing {arch} | Fold {fold} | Rank {rank}")

                # Load Model
                # pretrained=False is faster as we overwrite weights immediately
                model = get_model(arch, config, pretrained=False)

                try:
                    state_dict = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(state_dict)
                except Exception as e:
                    print(f"Error loading {checkpoint_path}: {e}")
                    continue

                model.eval()

                # Run inference on all TTA views for this model
                with torch.no_grad():
                    for loader in tta_loaders:
                        preds_list = []
                        for images, _ in loader:
                            images = images.to(device)
                            outputs = model(images)
                            probs = torch.sigmoid(outputs)
                            preds_list.append(probs.cpu().numpy())

                        # Concatenate batches for this view
                        batch_probs = np.concatenate(preds_list, axis=0)

                        # Accumulate
                        ensemble_probs += batch_probs
                        model_count += 1

    # 5. Finalize Predictions
    if model_count == 0:
        print("Warning: No valid checkpoints found! Outputting zeros.")
        final_probs = ensemble_probs  # All zeros
    else:
        # Average the accumulated probabilities
        final_probs = ensemble_probs / model_count
        print(
            f"Ensemble composed of {model_count} prediction vectors (Models x TTA Views)."
        )

    # 6. Format Submission
    print("Generating submission file...")
    submission_rows = []

    for idx, rec_id in enumerate(rec_ids):
        probs = final_probs[idx]
        for species_idx, p in enumerate(probs):
            # Construct Id: rec_id * 100 + species_id
            # Example: rec_id=1, species=2 -> Id=102
            row_id = int(rec_id * 100 + species_idx)
            submission_rows.append({"Id": row_id, "Probability": p})

    submission_df = pd.DataFrame(submission_rows)

    # Sort by Id to match sample submission convention
    submission_df = submission_df.sort_values(by="Id")

    # Save to disk
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
