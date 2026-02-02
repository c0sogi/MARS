import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.data_loader import get_dataloaders
from library.model_factory import get_model
from library.trainer import predict
from library.utils import set_seed, load_checkpoint


def predict_with_tta(loader, model, device, tta_steps=1):
    """
    Generates predictions using Test-Time Augmentation.
    Wraps the library.trainer.predict function which implements:
    1. Original image
    2. Horizontal Flip (if tta_steps >= 2)
    3. Vertical Flip (if tta_steps >= 3)

    Args:
        loader (DataLoader): The test data loader.
        model (nn.Module): The trained model.
        device (str): Device to run inference on.
        tta_steps (int): Number of TTA steps to perform.

    Returns:
        np.ndarray: Averaged predictions of shape (N, 1).
    """
    return predict(loader, model, device, tta_steps=tta_steps)


def run_inference(config: Config):
    """
    Main inference routine.
    Loads all available model checkpoints (across architectures and folds),
    generates TTA predictions, averages them (ensemble), and saves the submission.

    Args:
        config (Config): Configuration object.
    """
    # 1. Setup
    set_seed(config.seed)
    device = config.device
    print(f"Initializing Inference on {device}...")

    # 2. Load Test Data
    # We need the loader for images and the dataset for IDs
    test_loader = get_dataloaders(config, mode="test")
    test_ids = test_loader.dataset.image_ids

    print(f"Test set size: {len(test_ids)} images")

    # 3. Iterate over Ensemble Members
    ensemble_preds = []
    models_found = 0

    for model_name in config.model_names:
        for fold_id in range(config.n_folds):
            # Construct expected checkpoint path
            ckpt_filename = f"{model_name}_fold{fold_id}.pth"
            ckpt_path = os.path.join(config.output_dir, ckpt_filename)

            if not os.path.exists(ckpt_path):
                # This allows inference to run even if only partial folds/models are trained
                print(f"Checkpoint not found: {ckpt_filename}. Skipping.")
                continue

            print(f"Processing {model_name} (Fold {fold_id})...")

            # Instantiate model structure
            # We set pretrained=False because we are loading our own weights immediately
            model = get_model(
                model_name,
                num_classes=config.num_classes,
                pretrained=False,
                stem_surgery=config.stem_surgery,
            )

            # Load weights
            model = load_checkpoint(model, ckpt_path, device=device)
            model = model.to(device)

            # Generate predictions with TTA
            # shape: (N, 1)
            preds = predict_with_tta(
                test_loader, model, device, tta_steps=config.tta_steps
            )
            ensemble_preds.append(preds)
            models_found += 1

    if models_found == 0:
        print(
            "Error: No trained models found in output directory. Cannot generate submission."
        )
        return

    print(f"Aggregating predictions from {models_found} models...")

    # 4. Ensemble Aggregation (Soft Voting / Averaging)
    # Stack predictions: (Num_Models, N, 1)
    ensemble_preds = np.stack(ensemble_preds, axis=0)

    # Mean across models: (N, 1)
    avg_preds = np.mean(ensemble_preds, axis=0)

    # Flatten to (N,)
    avg_preds = avg_preds.flatten()

    # 5. Generate Submission File
    df_submission = pd.DataFrame({"id": test_ids, "has_cactus": avg_preds})

    # Ensure submission directory exists
    os.makedirs(config.submission_dir, exist_ok=True)

    # Save
    df_submission.to_csv(config.submission_path, index=False)
    print(f"Submission saved successfully to: {config.submission_path}")
    print("Head of submission:")
    print(df_submission.head())
