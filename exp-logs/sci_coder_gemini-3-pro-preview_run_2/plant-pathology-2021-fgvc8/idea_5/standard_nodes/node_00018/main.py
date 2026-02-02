import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import cv2
from library.config import Config
from library.utils import seed_everything, calculate_f1_score
from library.data import get_loaders, load_metadata
from library.models import AppleDiseaseModel
from library.engine import run_training
from library.inference import load_model_for_inference, generate_submission


def main():
    # 1. Configuration & Setup
    # Override epochs for fast baseline execution while ensuring convergence
    Config.EPOCHS = 4

    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Starting run with device: {device}")
    print(f"Training for {Config.EPOCHS} epochs per model.")

    # 2. Data Loading
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Train Model 1: ConvNeXt Large
    print(f"\n=== Training Model 1: {Config.MODEL_1_NAME} ===")
    model_1 = AppleDiseaseModel(Config.MODEL_1_NAME, pretrained=True)
    model_1.to(device)

    optimizer_1 = torch.optim.AdamW(
        model_1.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_1, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    run_training(
        model_1,
        train_loader,
        val_loader,
        optimizer_1,
        scheduler_1,
        device,
        Config.EPOCHS,
    )

    # Rename checkpoint to avoid overwriting by the next model
    src_ckpt = os.path.join(Config.WORKING_DIR, "model_best.pth")
    ckpt_1_path = os.path.join(Config.WORKING_DIR, "convnext_best.pth")
    if os.path.exists(src_ckpt):
        shutil.move(src_ckpt, ckpt_1_path)
        print(f"Saved Model 1 checkpoint to {ckpt_1_path}")
    else:
        print("Model 1 training failed to save checkpoint.")
        return

    # Free memory
    del model_1, optimizer_1, scheduler_1
    torch.cuda.empty_cache()

    # 4. Train Model 2: SwinV2 Large
    print(f"\n=== Training Model 2: {Config.MODEL_2_NAME} ===")
    model_2 = AppleDiseaseModel(Config.MODEL_2_NAME, pretrained=True)
    model_2.to(device)

    optimizer_2 = torch.optim.AdamW(
        model_2.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_2, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    run_training(
        model_2,
        train_loader,
        val_loader,
        optimizer_2,
        scheduler_2,
        device,
        Config.EPOCHS,
    )

    # Rename checkpoint
    src_ckpt = os.path.join(Config.WORKING_DIR, "model_best.pth")
    ckpt_2_path = os.path.join(Config.WORKING_DIR, "swin_best.pth")
    if os.path.exists(src_ckpt):
        shutil.move(src_ckpt, ckpt_2_path)
        print(f"Saved Model 2 checkpoint to {ckpt_2_path}")
    else:
        print("Model 2 training failed to save checkpoint.")
        return

    del model_2, optimizer_2, scheduler_2
    torch.cuda.empty_cache()

    # 5. Ensemble Validation
    print("\n=== Running Ensemble Validation ===")
    # Load models
    m1 = load_model_for_inference(Config.MODEL_1_NAME, ckpt_1_path, device)
    m2 = load_model_for_inference(Config.MODEL_2_NAME, ckpt_2_path, device)

    if m1 is None or m2 is None:
        print("Failed to load models for validation.")
        return

    models = [m1, m2]

    val_f1, val_preds_probs, val_targets = validate_ensemble(models, val_loader, device)

    print(f"Final Validation Metric: {val_f1}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    analyze_failures(val_preds_probs, val_targets, val_loader)

    # 7. Submission
    THRESHOLD = 0.9228752356223593
    if val_f1 > THRESHOLD:
        print(
            f"\nValidation Score ({val_f1}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        generate_submission(
            [(Config.MODEL_1_NAME, ckpt_1_path), (Config.MODEL_2_NAME, ckpt_2_path)]
        )
    else:
        print(
            f"\nValidation Score ({val_f1}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


def validate_ensemble(models, loader, device):
    """
    Runs inference on validation set with TTA and Ensemble logic.
    Returns F1 score, probabilities, and targets.
    """
    for m in models:
        m.eval()

    all_preds = []
    all_targets = []

    use_tta = Config.USE_TTA

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            batch_size = images.size(0)

            # TTA Inputs
            inputs = [images]
            if use_tta:
                inputs.append(torch.flip(images, dims=[3]))  # Horizontal flip

            model_probabilities = []

            for model in models:
                view_preds = []
                for inp in inputs:
                    with torch.cuda.amp.autocast(enabled=True):
                        logits = model(inp)
                        probs = torch.sigmoid(logits)
                    view_preds.append(probs)

                # Average TTA views
                avg_view_preds = torch.stack(view_preds).mean(dim=0)
                model_probabilities.append(avg_view_preds)

            # Stack models: (NumModels, B, C)
            all_model_preds = torch.stack(model_probabilities)

            # NaN-Safe Aggregation
            final_batch_probs = []
            for i in range(batch_size):
                sample_preds = all_model_preds[:, i, :]
                valid_preds = []
                for m_idx in range(sample_preds.shape[0]):
                    p = sample_preds[m_idx]
                    if not (torch.isnan(p).any() or torch.isinf(p).any()):
                        valid_preds.append(p)

                if valid_preds:
                    ensemble_p = torch.stack(valid_preds).mean(dim=0)
                else:
                    ensemble_p = torch.zeros(sample_preds.shape[1], device=device)

                final_batch_probs.append(ensemble_p)

            batch_probs = torch.stack(final_batch_probs).cpu().numpy()
            all_preds.append(batch_probs)
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    score = calculate_f1_score(all_targets, all_preds)
    return score, all_preds, all_targets


def analyze_failures(preds, targets, loader):
    """
    Correlates error magnitude with image metadata.
    """
    # Calculate Mean Absolute Error per sample
    errors = np.abs(preds - targets).mean(axis=1)

    # Get Image IDs
    image_ids = []
    for _, _, ids in loader:
        image_ids.extend(ids)

    # Load Metadata
    try:
        df = load_metadata("val")
        # Align with loader order
        df = df.set_index("image").loc[image_ids].reset_index()

        df["error"] = errors

        # Extract features
        file_sizes = []
        widths = []
        heights = []

        print("Extracting metadata features...")
        for _, row in df.iterrows():
            path = os.path.join(Config.INPUT_DIR, row["file_path"])
            try:
                file_sizes.append(os.path.getsize(path))
                # Quick read for dims
                img = cv2.imread(path)
                if img is not None:
                    h, w, _ = img.shape
                    widths.append(w)
                    heights.append(h)
                else:
                    widths.append(0)
                    heights.append(0)
            except:
                file_sizes.append(0)
                widths.append(0)
                heights.append(0)

        df["file_size"] = file_sizes
        df["width"] = widths
        df["height"] = heights

        print("Correlation between Error Magnitude and Features:")
        for feat in ["file_size", "width", "height"]:
            if df[feat].std() > 0:
                corr = df["error"].corr(df[feat])
                print(f"{feat}: {corr}")
            else:
                print(f"{feat}: NaN (No variance)")

    except Exception as e:
        print(f"Failure analysis failed: {e}")


if __name__ == "__main__":
    main()
