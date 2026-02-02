import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.data import get_loaders
from library.models import create_model
from library.utils import load_checkpoint


def predict_with_tta(model, loader, device):
    """
    Performs inference with Test-Time Augmentation (TTA).
    Averages predictions from: Original, Horizontal Flip, Vertical Flip.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            ids = batch["image_id"]

            # 1. Original Prediction
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            if Config.USE_TTA:
                # 2. Horizontal Flip
                # Images are (B, C, H, W). Flip on W (dim 3)
                images_h = torch.flip(images, dims=[3])
                outputs_h = model(images_h)
                probs_h = torch.softmax(outputs_h, dim=1)

                # 3. Vertical Flip
                # Flip on H (dim 2)
                images_v = torch.flip(images, dims=[2])
                outputs_v = model(images_v)
                probs_v = torch.softmax(outputs_v, dim=1)

                # Average probabilities
                probs = (probs + probs_h + probs_v) / 3.0

            all_preds.append(probs.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_preds), all_ids


def run_inference(debug=False):
    """
    Main inference routine.
    Loads both models, performs TTA, averages predictions, and saves submission.
    """
    device = Config.DEVICE
    print(f"Starting inference on device: {device}")

    # Storage for ensemble predictions
    ensemble_preds = []
    final_image_ids = None

    # ==========================================
    # Model 1: High-Resolution Expert (EfficientNet-B3)
    # ==========================================
    print(
        f"Processing Model 1: {Config.MODEL_1_NAME} (Size: {Config.MODEL_1_IMG_SIZE})"
    )

    # Determine checkpoint path (SWA > Best)
    swa_path_1 = os.path.join(Config.WORK_DIR, f"swa_model_{Config.MODEL_1_NAME}.pth")
    best_path_1 = os.path.join(Config.WORK_DIR, f"best_model_{Config.MODEL_1_NAME}.pth")
    ckpt_path_1 = swa_path_1 if os.path.exists(swa_path_1) else best_path_1

    if os.path.exists(ckpt_path_1):
        print(f"Loading checkpoint: {ckpt_path_1}")
        # Initialize model
        model_1 = create_model(
            Config.MODEL_1_NAME, num_classes=Config.NUM_CLASSES, pretrained=False
        )
        model_1.to(device)

        # Load weights
        load_checkpoint(ckpt_path_1, model_1)

        # Get Data Loader (Test set only)
        # Note: get_loaders returns (train, val, test)
        _, _, test_loader_1 = get_loaders(
            Config.MODEL_1_IMG_SIZE, batch_size=Config.BATCH_SIZE, debug=debug
        )

        # Predict
        preds_1, ids_1 = predict_with_tta(model_1, test_loader_1, device)
        ensemble_preds.append(preds_1)
        final_image_ids = ids_1

        # Cleanup
        del model_1, test_loader_1
        torch.cuda.empty_cache()
    else:
        print(f"Warning: Checkpoint for {Config.MODEL_1_NAME} not found. Skipping.")

    # ==========================================
    # Model 2: Contextual Expert (ConvNeXt-Tiny)
    # ==========================================
    print(
        f"Processing Model 2: {Config.MODEL_2_NAME} (Size: {Config.MODEL_2_IMG_SIZE})"
    )

    swa_path_2 = os.path.join(Config.WORK_DIR, f"swa_model_{Config.MODEL_2_NAME}.pth")
    best_path_2 = os.path.join(Config.WORK_DIR, f"best_model_{Config.MODEL_2_NAME}.pth")
    ckpt_path_2 = swa_path_2 if os.path.exists(swa_path_2) else best_path_2

    if os.path.exists(ckpt_path_2):
        print(f"Loading checkpoint: {ckpt_path_2}")
        model_2 = create_model(
            Config.MODEL_2_NAME, num_classes=Config.NUM_CLASSES, pretrained=False
        )
        model_2.to(device)

        load_checkpoint(ckpt_path_2, model_2)

        _, _, test_loader_2 = get_loaders(
            Config.MODEL_2_IMG_SIZE, batch_size=Config.BATCH_SIZE, debug=debug
        )

        preds_2, ids_2 = predict_with_tta(model_2, test_loader_2, device)

        # Verify ID alignment
        if final_image_ids is not None and ids_2 != final_image_ids:
            raise ValueError("Mismatch in test set image IDs between models.")
        final_image_ids = ids_2

        ensemble_preds.append(preds_2)

        del model_2, test_loader_2
        torch.cuda.empty_cache()
    else:
        print(f"Warning: Checkpoint for {Config.MODEL_2_NAME} not found. Skipping.")

    # ==========================================
    # Ensemble & Submission
    # ==========================================
    if not ensemble_preds:
        raise RuntimeError(
            "No models were successfully loaded. Cannot generate submission."
        )

    # Average predictions
    print(f"Ensembling predictions from {len(ensemble_preds)} models...")
    final_probs = np.mean(ensemble_preds, axis=0)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(final_probs, columns=Config.CLASSES)
    submission_df.insert(0, "image_id", final_image_ids)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
