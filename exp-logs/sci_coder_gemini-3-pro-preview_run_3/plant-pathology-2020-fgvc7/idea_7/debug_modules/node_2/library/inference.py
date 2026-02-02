import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.models import AppleEfficientNet, AppleSwin
from library.data import get_test_loader


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (TTA).
    TTA Strategy: Original, Horizontal Flip, Vertical Flip.
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Original
            out = model(images)
            probs = torch.softmax(out, dim=1)

            # 2. Horizontal Flip
            images_h = torch.flip(images, dims=[3])
            out_h = model(images_h)
            probs_h = torch.softmax(out_h, dim=1)

            # 3. Vertical Flip
            images_v = torch.flip(images, dims=[2])
            out_v = model(images_v)
            probs_v = torch.softmax(out_v, dim=1)

            # Average predictions
            avg_probs = (probs + probs_h + probs_v) / 3.0
            preds_list.append(avg_probs.cpu().numpy())

    return np.concatenate(preds_list, axis=0)


def run_inference():
    """
    Orchestrates the inference pipeline:
    1. Loads test metadata.
    2. Iterates over 5 folds and 2 model architectures (EffNet, Swin).
    3. Aggregates predictions.
    4. Saves submission file.
    """
    print("Starting Inference Pipeline...")
    device = Config.DEVICE

    # Load test metadata to get IDs
    test_df = pd.read_csv(Config.TEST_CSV)
    image_ids = test_df["image_id"].values
    num_samples = len(test_df)
    num_classes = Config.NUM_CLASSES

    # Accumulator for predictions
    final_preds = np.zeros((num_samples, num_classes), dtype=np.float32)
    model_count = 0

    # Iterate over folds
    for fold in range(Config.N_FOLDS):
        # ---------------------------
        # EfficientNet-B4
        # ---------------------------
        effnet_path = os.path.join(Config.WORKING_DIR, f"effnet_fold_{fold}_best.pth")
        if os.path.exists(effnet_path):
            print(f"Processing Fold {fold} | Model: EfficientNet-B4")

            # Initialize model (pretrained=False to avoid download during inference)
            model = AppleEfficientNet(pretrained=False)
            model.load_state_dict(torch.load(effnet_path, map_location=device))
            model.to(device)

            # Get loader with specific image size
            loader = get_test_loader(Config.EFFNET_IMG_SIZE, Config.BATCH_SIZE)

            # Predict
            preds = predict_with_tta(model, loader, device)
            final_preds += preds
            model_count += 1

            # Cleanup
            del model, loader, preds
            torch.cuda.empty_cache()
        else:
            print(f"Warning: Checkpoint not found at {effnet_path}")

        # ---------------------------
        # Swin Transformer Small
        # ---------------------------
        swin_path = os.path.join(Config.WORKING_DIR, f"swin_fold_{fold}_best.pth")
        if os.path.exists(swin_path):
            print(f"Processing Fold {fold} | Model: Swin-Small")

            # Initialize model
            model = AppleSwin(pretrained=False)
            model.load_state_dict(torch.load(swin_path, map_location=device))
            model.to(device)

            # Get loader
            loader = get_test_loader(Config.SWIN_IMG_SIZE, Config.BATCH_SIZE)

            # Predict
            preds = predict_with_tta(model, loader, device)
            final_preds += preds
            model_count += 1

            # Cleanup
            del model, loader, preds
            torch.cuda.empty_cache()
        else:
            print(f"Warning: Checkpoint not found at {swin_path}")

    # Normalize predictions
    if model_count > 0:
        final_preds /= model_count
        print(f"Inference complete. Aggregated {model_count} models.")
    else:
        print("Error: No models were found/loaded. Generating uniform predictions.")
        final_preds = np.full((num_samples, num_classes), 1.0 / num_classes)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(final_preds, columns=Config.CLASS_LABELS)
    submission_df.insert(0, "image_id", image_ids)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")

    # Print first few rows for verification
    print("\nSubmission Head:")
    print(submission_df.head())
