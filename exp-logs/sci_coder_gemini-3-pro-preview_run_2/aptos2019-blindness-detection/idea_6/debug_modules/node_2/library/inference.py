import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.model import DRModel
from library.data import load_dataframe, RetinopathyDataset, get_transforms


def predict_ensemble(load_cached_data=True):
    """
    Generates predictions for the test set using an ensemble of trained models.
    Applies Test-Time Augmentation (TTA) and averages predictions.

    Args:
        load_cached_data (bool): Whether to attempt loading cached DataFrames.
    """
    device = Config.DEVICE

    # Ensure output directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Load Test Metadata
    # We use the library function which handles caching logic
    test_df = load_dataframe(Config.TEST_CSV, "test_df", load_cached_data)
    id_codes = test_df["id_code"].values

    # Initialize accumulator for predictions
    # We sum continuous scores from all models and divide at the end
    ensemble_preds = np.zeros(len(test_df), dtype=np.float32)
    model_count = 0

    # 2. Define Ensemble Configurations
    # We iterate over model types because they require different image resolutions
    ensemble_configs = [
        (Config.MODEL_CNN, Config.NUM_FOLDS),
        (Config.MODEL_TRANS, Config.NUM_FOLDS),
    ]

    print(f"Starting inference on {len(test_df)} test images...")

    for model_cfg, num_folds in ensemble_configs:
        img_size = model_cfg["img_size"]
        model_name = model_cfg["name"]
        prefix = model_cfg["checkpoint_prefix"]
        batch_size = model_cfg["batch_size"]

        # 3. Create DataLoader for specific resolution
        # Use 'val' transforms which only do Resize + Normalize (no random augs)
        transforms = get_transforms(img_size, mode="val")

        dataset = RetinopathyDataset(
            test_df, transforms=transforms, mode="test", input_dir=Config.INPUT_DIR
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 4. Iterate over folds
        for fold in range(num_folds):
            # Prioritize SWA checkpoint, fallback to Best checkpoint
            swa_path = os.path.join(Config.WORKING_DIR, f"{prefix}_fold_{fold}_swa.pth")
            best_path = os.path.join(
                Config.WORKING_DIR, f"{prefix}_fold_{fold}_best.pth"
            )

            checkpoint_path = None
            if os.path.exists(swa_path):
                checkpoint_path = swa_path
            elif os.path.exists(best_path):
                checkpoint_path = best_path

            if checkpoint_path is None:
                # If training was incomplete or skipped for this fold, we skip inference
                continue

            # Load Model
            model = DRModel(model_name=model_name, pretrained=False)
            state_dict = torch.load(checkpoint_path, map_location=device)

            # Fix for SWA (AveragedModel) or DataParallel checkpoints which add 'module.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                elif k == "n_averaged":
                    continue
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
            model.to(device)
            model.eval()

            fold_preds = []

            # 5. Inference Loop with TTA
            with torch.no_grad():
                for images in loader:
                    images = images.to(device)

                    # Forward Pass 1: Original
                    out_orig = model(images)

                    # Forward Pass 2: Horizontal Flip (TTA)
                    # Flip along width dimension (dim 3 for B,C,H,W)
                    images_flip = torch.flip(images, dims=[3])
                    out_flip = model(images_flip)

                    # Average predictions (Soft Voting)
                    batch_preds = (out_orig + out_flip) / 2.0
                    fold_preds.append(batch_preds.cpu().numpy())

            # Accumulate
            fold_preds_flat = np.concatenate(fold_preds).flatten()
            ensemble_preds += fold_preds_flat
            model_count += 1

            # Cleanup to save memory
            del model
            torch.cuda.empty_cache()

    # 6. Finalize Predictions
    if model_count > 0:
        # Average across all models
        avg_preds = ensemble_preds / model_count

        # Round to nearest integer and clip to valid range [0, 4]
        final_preds = np.rint(avg_preds).clip(0, 4).astype(int)
    else:
        # Fallback if no models were found (e.g. debugging flow)
        print("Warning: No trained models found. Predicting all zeros.")
        final_preds = np.zeros(len(test_df), dtype=int)

    # 7. Create Submission File
    submission = pd.DataFrame({"id_code": id_codes, "diagnosis": final_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Inference complete. Processed {model_count} models.")
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
