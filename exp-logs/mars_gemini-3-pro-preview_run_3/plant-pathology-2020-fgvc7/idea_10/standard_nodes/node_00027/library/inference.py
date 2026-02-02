import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything
from library.data import get_test_loader
from library.models import HeterogeneousExpert


def predict_with_tta(debug=False):
    """
    Performs inference on the test set using Domain-Aware Test-Time Augmentation (TTA).
    Averages predictions from original and horizontally flipped images across all
    trained models and folds. Generates the final submission file.

    Args:
        debug (bool): If True, processes only a few batches for debugging purposes.
    """
    seed_everything(Config.seed)

    # Load Test Metadata
    test_meta_path = os.path.join(Config.metadata_dir, "test.csv")
    if not os.path.exists(test_meta_path):
        raise FileNotFoundError(f"Test metadata file not found at {test_meta_path}")

    test_df = pd.read_csv(test_meta_path)
    image_ids = test_df["image_id"].values

    # Initialize accumulator for ensemble predictions
    # Shape: (N_samples, N_classes)
    final_preds = np.zeros((len(test_df), Config.num_classes), dtype=np.float32)
    model_count = 0

    # Track number of samples processed (for debug handling)
    num_samples_processed = len(test_df)

    print(f"Starting Inference (TTA Enabled)... Total Images: {len(test_df)}")

    # Iterate over defined model configurations
    for backbone_name, img_size in Config.models_config:
        print(f"\nProcessing Backbone: {backbone_name} (Input Size: {img_size})")

        # Prepare DataLoaders
        # 1. Original Images
        loader_orig = get_test_loader(
            img_size=img_size,
            batch_size=Config.batch_size,
            num_workers=Config.num_workers,
            tta=False,
        )

        # 2. Horizontally Flipped Images (TTA)
        loader_flip = get_test_loader(
            img_size=img_size,
            batch_size=Config.batch_size,
            num_workers=Config.num_workers,
            tta=True,
        )

        # Iterate over folds
        for fold in range(Config.n_folds):
            model_filename = f"{backbone_name.replace('.', '_')}_fold_{fold}.pth"
            model_path = os.path.join(Config.working_dir, model_filename)

            if not os.path.exists(model_path):
                print(
                    f"  [Fold {fold}] Model file not found: {model_filename}. Skipping."
                )
                continue

            print(f"  [Fold {fold}] Evaluating...")

            # Initialize Model
            # pretrained=False because we are loading custom weights
            model = HeterogeneousExpert(
                backbone_name=backbone_name,
                num_classes=Config.num_classes,
                pretrained=False,
            )
            model.to(Config.device)

            # Load Weights
            state_dict = torch.load(model_path, map_location=Config.device)
            model.load_state_dict(state_dict)
            model.eval()

            # Helper function for inference loop
            def run_inference_loop(loader):
                preds = []
                with torch.no_grad():
                    for i, (images, _) in enumerate(loader):
                        if debug and i >= 2:
                            break
                        images = images.to(Config.device)
                        outputs = model(images)
                        # Apply Softmax to get probabilities
                        probs = torch.softmax(outputs, dim=1)
                        preds.append(probs.cpu().numpy())
                return np.concatenate(preds)

            # Predict on Original
            preds_orig = run_inference_loop(loader_orig)

            # Predict on Flip
            preds_flip = run_inference_loop(loader_flip)

            # Average TTA (Original + Flip)
            fold_preds = (preds_orig + preds_flip) / 2.0

            # Update global accumulator
            # Handle debug case where fold_preds might be smaller than final_preds
            current_len = len(fold_preds)
            final_preds[:current_len] += fold_preds
            num_samples_processed = current_len

            model_count += 1

            # Cleanup to free GPU memory
            del model, state_dict
            torch.cuda.empty_cache()

    # Average over all models
    if model_count > 0:
        final_preds /= model_count
        print(f"\nEnsemble complete. Averaged over {model_count} models.")
    else:
        print("\nWarning: No models were found or processed. Predictions are zeros.")

    # Slice results if in debug mode
    final_preds = final_preds[:num_samples_processed]
    image_ids = image_ids[:num_samples_processed]

    # Create Submission DataFrame
    submission = pd.DataFrame(final_preds, columns=Config.target_cols)
    submission.insert(0, "image_id", image_ids)

    # Save Submission
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print("First 5 rows:")
    print(submission.head())
