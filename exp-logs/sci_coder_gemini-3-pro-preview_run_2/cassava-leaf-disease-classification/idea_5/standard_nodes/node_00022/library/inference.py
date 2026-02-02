import os
import pandas as pd
import torch
import torch.nn.functional as F

from library.config import Config
from library.utils import seed_everything
from library.data import get_test_dataloader
from library.model import get_model


def ensemble_inference(cfg):
    """
    Performs inference using an ensemble of models from 5 folds with Test Time Augmentation (TTA).
    Loads saved checkpoints, predicts on the test set, averages probabilities, and saves the submission.

    Args:
        cfg (Config): Configuration object containing paths, hyperparameters, and device settings.
    """
    # Ensure reproducibility
    seed_everything(cfg.seed)

    device = cfg.device
    print(f"Starting Ensemble Inference on device: {device}")

    # 1. Prepare Data
    # Load test dataloader. Note: shuffle=False is guaranteed by get_test_dataloader
    test_loader = get_test_dataloader(cfg)

    # Get number of samples to initialize storage
    # Accessing the underlying dataset to get the total count
    num_test_samples = len(test_loader.dataset)

    # Tensor to store accumulated probabilities [N_samples, N_classes]
    ensemble_probs = torch.zeros((num_test_samples, cfg.num_classes), device=device)

    models_found = 0

    # 2. Iterate through each fold
    for fold in range(cfg.n_folds):
        # Construct path to the best model for this fold
        model_path = os.path.join(cfg.working_dir, f"fold_{fold}_best.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model checkpoint not found at {model_path}. Skipping Fold {fold}."
            )
            continue

        print(f"Loading model for Fold {fold} from {model_path}...")

        # Initialize model architecture
        # pretrained=False because we are loading our own trained weights
        model = get_model(cfg, pretrained=False)

        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        fold_probs_list = []

        # 3. Inference Loop
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)

                # Forward Pass (Original)
                logits = model(images)

                # Test Time Augmentation (TTA): Horizontal Flip
                if cfg.tta:
                    # Flip the width dimension (dim 3)
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flipped = model(images_flipped)

                    # Average logits
                    logits = (logits + logits_flipped) / 2.0

                # Convert logits to probabilities via Softmax
                probs = F.softmax(logits, dim=1)
                fold_probs_list.append(probs)

        # Concatenate batches for this fold
        fold_probs = torch.cat(fold_probs_list, dim=0)

        # Accumulate to ensemble
        ensemble_probs += fold_probs
        models_found += 1

        # Clean up memory
        del model, checkpoint, fold_probs, fold_probs_list
        torch.cuda.empty_cache()

    # 4. Aggregate Predictions
    if models_found > 0:
        ensemble_probs /= models_found
    else:
        print("Error: No models were loaded. Predictions will default to class 0.")

    # Get final class predictions (Argmax)
    final_preds = torch.argmax(ensemble_probs, dim=1).cpu().numpy()

    # 5. Generate Submission File
    # Load test metadata to get correct image_ids
    df_test = pd.read_csv(cfg.test_metadata_path)

    submission_df = pd.DataFrame(
        {"image_id": df_test["image_id"], "label": final_preds}
    )

    # Ensure submission directory exists
    os.makedirs(cfg.submission_dir, exist_ok=True)
    save_path = os.path.join(cfg.submission_dir, "submission.csv")

    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved successfully to {save_path}")
    print("First 5 predictions:")
    print(submission_df.head())
