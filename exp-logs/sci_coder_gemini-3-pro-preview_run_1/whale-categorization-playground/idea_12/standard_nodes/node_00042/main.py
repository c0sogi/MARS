import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2

# Import from library
from library.config import Config
from library.utils import seed_everything, calculate_map5
from library.dataset import get_dataloaders
from library.models import WhaleModel
from library.trainer import Trainer
from library.inference import predict_ensemble, load_ensemble
from library.loss import ArcFaceLoss


def run():
    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Baseline
    # -------------------------------------------------------------------------
    # The task requires a fast baseline execution (< 2 hours).
    # We reduce epochs to ensure completion while maintaining the ensemble strategy.
    Config.MAX_EPOCHS = 8
    Config.EARLY_STOPPING_PATIENCE = 3

    print("Starting runfile.py execution...")
    print(
        f"Configuration: Max Epochs={Config.MAX_EPOCHS}, Patience={Config.EARLY_STOPPING_PATIENCE}"
    )

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading datasets...")
    # Load cached data to ensure consistency and speed
    train_loader, val_loader, classes = get_dataloaders(load_cached_data=True)
    num_classes = len(classes)
    device = Config.DEVICE
    print(f"Device: {device}")
    print(f"Number of classes: {num_classes}")

    # -------------------------------------------------------------------------
    # 3. Ensemble Training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("STARTING ENSEMBLE TRAINING")
    print("=" * 40)

    for model_cfg in Config.ENSEMBLE_MODELS:
        arch = model_cfg["arch"]
        seed = model_cfg["seed"]
        name = model_cfg["name"]

        print(f"\n--- Training Model: {name} (Arch: {arch}, Seed: {seed}) ---")

        # Set seed for this specific model training
        seed_everything(seed)

        # Initialize Model
        # pretrained=True to leverage ImageNet features for faster convergence
        model = WhaleModel(arch=arch, num_classes=num_classes, pretrained=True)
        model.to(device)

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            device=device,
            model_name=name,
        )

        # Train
        best_score = trainer.train_until_convergence()
        print(f"Finished training {name}. Best Validation Score: {best_score}")

        # Cleanup to free GPU memory for next model
        del model, trainer
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Ensemble Validation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("ENSEMBLE VALIDATION")
    print("=" * 40)

    seed_everything(42)  # Reset seed for deterministic validation

    # Load all trained models
    models, criterions = load_ensemble(device, num_classes)

    if not models:
        print("Error: No models loaded. Exiting.")
        return

    all_preds = []
    all_targets = []
    all_image_paths = []

    # Access dataframe to get file paths for failure analysis
    val_df = val_loader.dataset.df

    print("Running inference on validation set...")

    with torch.no_grad():
        batch_idx = 0
        for images, labels in val_loader:
            images = images.to(device)
            batch_size = images.size(0)

            # Accumulator for ensemble logits
            ensemble_logits = torch.zeros((batch_size, num_classes), device=device)

            for i in range(len(models)):
                model = models[i]
                criterion = criterions[i]

                # Get class centers
                class_centers = F.normalize(criterion.weight, p=2, dim=1)

                # View 1: Original
                emb = model(images)
                emb_norm = F.normalize(emb, p=2, dim=1)
                logits = F.linear(emb_norm, class_centers)

                # View 2: Flip (TTA)
                if Config.TTA_FLIP:
                    images_flip = torch.flip(images, dims=[3])
                    emb_flip = model(images_flip)
                    emb_flip_norm = F.normalize(emb_flip, p=2, dim=1)
                    logits_flip = F.linear(emb_flip_norm, class_centers)
                    logits = (logits + logits_flip) / 2.0

                ensemble_logits += logits

            # Average logits
            ensemble_logits /= len(models)

            # Get Top 5
            _, top_indices = torch.topk(ensemble_logits, k=5, dim=1)

            # Store results
            all_preds.extend(top_indices.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())

            # Store paths
            start_idx = batch_idx * Config.BATCH_SIZE
            end_idx = start_idx + batch_size
            # Handle last batch which might be smaller
            current_paths = val_df.iloc[start_idx : min(end_idx, len(val_df))][
                "file_path"
            ].values
            all_image_paths.extend(current_paths)

            batch_idx += 1

    # Calculate Final Metric
    final_metric = calculate_map5(all_preds, all_targets)
    # Print exactly as required
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    errors = []
    widths = []
    heights = []
    aspect_ratios = []
    intensities = []

    print("Analyzing error patterns...")

    for i, pred_row in enumerate(all_preds):
        target = all_targets[i]

        # Calculate score: 1/(rank+1) if present, else 0
        score = 0.0
        if target in pred_row:
            rank = list(pred_row).index(target)
            score = 1.0 / (rank + 1)

        # Error magnitude
        error = 1.0 - score
        errors.append(error)

        # Image Stats
        full_path = os.path.join(Config.INPUT_DIR, all_image_paths[i])

        # We use cv2 to quickly get dimensions and mean
        img = cv2.imread(full_path)
        if img is not None:
            h, w = img.shape[:2]
            # Simple mean intensity
            mean_val = np.mean(img) / 255.0

            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            intensities.append(mean_val)
        else:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            intensities.append(0)

    # Calculate Correlations
    if len(errors) > 1:
        # Helper for correlation
        def get_corr(a, b):
            if np.std(a) == 0 or np.std(b) == 0:
                return 0.0
            return np.corrcoef(a, b)[0, 1]

        c_width = get_corr(errors, widths)
        c_height = get_corr(errors, heights)
        c_ratio = get_corr(errors, aspect_ratios)
        c_int = get_corr(errors, intensities)

        print(f"Correlation (Error vs Width): {c_width:.6f}")
        print(f"Correlation (Error vs Height): {c_height:.6f}")
        print(f"Correlation (Error vs Aspect Ratio): {c_ratio:.6f}")
        print(f"Correlation (Error vs Intensity): {c_int:.6f}")
    else:
        print("Not enough samples for correlation analysis.")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("SUBMISSION GENERATION")
    print("=" * 40)

    THRESHOLD = 0.6636363636363634

    if final_metric > THRESHOLD:
        print(f"Validation Metric ({final_metric}) exceeds threshold ({THRESHOLD}).")
        print("Proceeding to generate submission...")

        # Clear memory
        del models, criterions, all_preds, all_targets, ensemble_logits
        torch.cuda.empty_cache()

        # Run Inference
        predict_ensemble()
    else:
        print(
            f"Validation Metric ({final_metric}) does not exceed threshold ({THRESHOLD})."
        )
        print("Submission generation skipped.")


if __name__ == "__main__":
    run()
