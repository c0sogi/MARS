import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.dataset import RetinopathyDataset, get_transforms, load_dataframe
from library.models import RetinopathyModel


def predict(load_cached_data: bool = True, output_path: str = None):
    """
    Runs inference on the test set using the ensemble of trained models.
    Performs Test-Time Augmentation (TTA) and aggregates predictions.

    Args:
        load_cached_data (bool): Whether to use cached Parquet files for metadata.
        output_path (str): Path to save the submission CSV. Defaults to Config.submission_path.
    """
    seed_everything(Config.seed)

    if output_path is None:
        output_path = Config.submission_path

    # 1. Load Test Data
    # load_dataframe handles caching logic internally
    test_df = load_dataframe(Config.test_csv_path, "test_df", load_cached_data)

    # Debug mode: subset data
    if Config.debug:
        test_df = test_df.head(Config.debug_samples)

    # Create Dataset and Loader
    # Mode='test' ensures __getitem__ returns only the image
    test_dataset = RetinopathyDataset(
        test_df, transform=get_transforms("test"), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 2. Ensemble Inference
    # We will accumulate continuous regression scores from all models
    num_samples = len(test_df)
    accumulated_preds = np.zeros(num_samples, dtype=np.float32)
    model_count = 0

    print(f"Starting inference on {num_samples} images...")

    # Iterate through all architectures and folds
    for arch in Config.model_archs:
        for fold in range(Config.n_folds):
            checkpoint_path = os.path.join(
                Config.working_dir, f"{arch}_fold_{fold}.pth"
            )

            # Skip if checkpoint doesn't exist (e.g., during partial debugging)
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint not found at {checkpoint_path}. Skipping.")
                continue

            print(f"Loading model: {arch} (Fold {fold})")

            # Initialize model and load weights
            # pretrained=False because we are loading our own weights
            model = RetinopathyModel(model_name=arch, pretrained=False)
            state_dict = torch.load(checkpoint_path, map_location=Config.device)
            model.load_state_dict(state_dict)
            model.to(Config.device)
            model.eval()

            fold_preds = []

            # Inference Loop
            with torch.no_grad():
                for images in test_loader:
                    images = images.to(Config.device)

                    # TTA Step 1: Original Image
                    out_orig = model(images).view(-1)

                    # TTA Step 2: Horizontal Flip
                    # images shape is (B, C, H, W), flip on last dimension
                    images_flipped = torch.flip(images, dims=[3])
                    out_flip = model(images_flipped).view(-1)

                    # Average the predictions
                    batch_preds = (out_orig + out_flip) / 2.0
                    fold_preds.append(batch_preds.cpu().numpy())

            # Accumulate results
            fold_preds = np.concatenate(fold_preds)
            accumulated_preds += fold_preds
            model_count += 1

            # Cleanup to free GPU memory
            del model, state_dict, fold_preds
            gc.collect()
            torch.cuda.empty_cache()

    if model_count == 0:
        raise RuntimeError(
            "No valid model checkpoints found. Cannot generate predictions."
        )

    # 3. Aggregation and Post-processing
    # Average over the ensemble
    avg_preds = accumulated_preds / model_count

    # Round to nearest integer for ordinal classification
    preds_rounded = np.round(avg_preds)

    # Clip to valid range [0, 4]
    preds_clipped = np.clip(preds_rounded, 0, 4).astype(int)

    # 4. Generate Submission File
    submission = pd.DataFrame(
        {"id_code": test_df["id_code"], "diagnosis": preds_clipped}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    return submission
