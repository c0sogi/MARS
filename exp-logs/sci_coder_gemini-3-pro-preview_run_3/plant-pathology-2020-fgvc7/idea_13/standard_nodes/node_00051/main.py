import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import timm
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config, resolve_model_name
from library.utils import seed_everything, get_device, calculate_metric, load_checkpoint
from library.dataset import load_data, get_class_weights, get_loaders, get_test_loader
from library.model import AppleMultiTaskModel
from library.loss import DecoupledMultiTaskLoss
from library.engine import train_model


def run_inference(model, loader, device, use_tta=False):
    """
    Runs inference on a loader using the given model.
    Supports Domain-Aware TTA (Horizontal Flip).
    Returns:
        probs (np.array): Shape (N, Num_Classes)
        ids (list): List of image IDs (if available in loader)
    """
    model.eval()
    all_probs = []
    all_ids = []

    # Check if loader returns IDs (Test loader does, Val loader does not usually)
    # Based on dataset.py: Val returns (image, targets), Test returns (image, image_id)

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                # Could be (img, target) or (img, id)
                # Test loader returns (image, image_id) where image_id is string/tuple
                # Val loader returns (image, targets_dict)
                elem2 = batch[1]
                if isinstance(elem2, dict):
                    # Validation Loader
                    images = batch[0]
                    ids = (
                        None  # IDs not strictly needed for val metrics if we have order
                    )
                else:
                    # Test Loader
                    images = batch[0]
                    ids = list(elem2)
            else:
                raise ValueError("Unexpected batch structure")

            images = images.to(device)

            # 1. Forward Pass (Original)
            out_orig = model(images)
            logits_orig = out_orig["main"]
            probs_orig = torch.softmax(logits_orig, dim=1)

            if use_tta and Config.TTA_FLIP_HORIZONTAL:
                # 2. Forward Pass (Horizontal Flip)
                images_flip = torch.flip(images, dims=[3])  # [B, C, H, W]
                out_flip = model(images_flip)
                logits_flip = out_flip["main"]
                probs_flip = torch.softmax(logits_flip, dim=1)

                # Average
                probs = (probs_orig + probs_flip) / 2.0
            else:
                probs = probs_orig

            all_probs.append(probs.cpu().numpy())
            if ids is not None:
                all_ids.extend(ids)

    return np.concatenate(all_probs), all_ids


def perform_failure_analysis(val_df, val_probs, val_targets):
    """
    Correlates model error with image meta-features.
    """
    print("\n==== Failure Analysis ====")

    # Calculate Error: 1.0 - Probability assigned to the true class
    # val_targets is one-hot
    true_class_indices = np.argmax(val_targets, axis=1)

    # Extract prob of true class
    # advanced indexing: [row_indices, col_indices]
    n_samples = len(val_probs)
    probs_of_true = val_probs[np.arange(n_samples), true_class_indices]
    errors = 1.0 - probs_of_true

    # Compute Image Stats
    # We need to read images again or assume we can do it fast enough
    print("Computing image meta-features for validation set...")
    brightness = []
    contrast = []

    for _, row in val_df.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness.append(np.mean(gray))
            contrast.append(np.std(gray))
        else:
            brightness.append(0)
            contrast.append(0)

    brightness = np.array(brightness)
    contrast = np.array(contrast)

    # Calculate Correlation
    corr_bright, _ = pearsonr(errors, brightness)
    corr_contrast, _ = pearsonr(errors, contrast)

    print(f"Correlation (Error vs Brightness): {corr_bright:.4f}")
    print(f"Correlation (Error vs Contrast):   {corr_contrast:.4f}")

    if abs(corr_bright) > 0.1:
        print("-> Observation: Model performance is sensitive to image brightness.")
    if abs(corr_contrast) > 0.1:
        print("-> Observation: Model performance is sensitive to image contrast.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()

    # Override Config for Fast Baseline
    Config.EPOCHS = 5
    print(f"Running Fast Baseline with EPOCHS={Config.EPOCHS}")

    # Cite debug_lesson_3: Query Registry Keys Programmatically.
    # Print available maxvit models to debug potential resolution errors.
    print(
        "\n[DEBUG] Available MaxViT models in registry:", timm.list_models("*maxvit*")
    )

    # Cite debug_lesson_6: Validate Registry Dependencies at Initialization
    print("\nValidating and Resolving Model Configurations...")
    for i, cfg in enumerate(Config.MODEL_CONFIGS):
        try:
            raw_name = cfg["backbone"]
            resolved_name = resolve_model_name(raw_name)
            print(
                f"  - Model '{cfg['name']}': Resolved '{raw_name}' -> '{resolved_name}'"
            )
            # Update the config in-place with the resolved name
            Config.MODEL_CONFIGS[i]["backbone"] = resolved_name
        except RuntimeError as e:
            print(f"  - Model '{cfg['name']}': FAILED to resolve '{raw_name}'")
            print(f"    Error: {e}")
            sys.exit(1)

    # 2. Data Loading
    train_df, val_df, test_df = load_data(debug=Config.DEBUG)
    class_weights = get_class_weights(load_cached_data=True)

    # Prepare storage for ensemble predictions
    # Shape: (N_samples, N_classes)
    val_ensemble_probs = np.zeros((len(val_df), Config.NUM_CLASSES))
    test_ensemble_probs = np.zeros((len(test_df), Config.NUM_CLASSES))

    trained_models_count = 0

    # 3. Iterate Models
    for model_cfg in Config.MODEL_CONFIGS:
        model_name = model_cfg["name"]
        backbone_name = model_cfg["backbone"]
        print(f"\n\nProcessing Model: {model_name} (Backbone: {backbone_name})")
        print("=" * 40)

        # Loaders
        train_loader, val_loader = get_loaders(
            train_df,
            val_df,
            img_size=model_cfg["img_size"],
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
        )

        test_loader = get_test_loader(
            test_df,
            img_size=model_cfg["img_size"],
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
        )

        # Initialize Model
        model = AppleMultiTaskModel(
            backbone_name=backbone_name,
            num_classes=Config.NUM_CLASSES,
            pretrained=True,
            gem_p=model_cfg["gem_p"],
            dropout=model_cfg["dropout"],
        ).to(device)

        # Loss
        loss_fn = DecoupledMultiTaskLoss(class_weights=class_weights, device=device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
        )

        # Train
        save_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")
        train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            device,
            Config,
            save_path,
            scheduler,
        )

        # Load Best Weights for Inference
        # We need to load into the model to ensure we use the best state
        load_checkpoint(save_path, model, device=device)

        # Inference (Val)
        print(f"Running Inference on Validation Set for {model_name}...")
        val_probs, _ = run_inference(model, val_loader, device, use_tta=True)
        val_ensemble_probs += val_probs

        # Inference (Test)
        print(f"Running Inference on Test Set for {model_name}...")
        test_probs, test_ids = run_inference(model, test_loader, device, use_tta=True)
        test_ensemble_probs += test_probs

        trained_models_count += 1

        # Free memory
        del model, optimizer, scheduler, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()

    # 4. Average Predictions
    val_ensemble_probs /= trained_models_count
    test_ensemble_probs /= trained_models_count

    # 5. Validation Metric
    # Get Ground Truth
    val_targets = val_df[Config.CLASS_LABELS].values
    final_metric = calculate_metric(val_targets, val_ensemble_probs)

    print(f"\nFinal Validation Metric: {final_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(val_df, val_ensemble_probs, val_targets)

    # 7. Submission
    # Create submission DataFrame
    # Columns: image_id, healthy, multiple_diseases, rust, scab
    submission_df = pd.DataFrame(test_ensemble_probs, columns=Config.CLASS_LABELS)
    submission_df.insert(0, "image_id", test_ids)

    # Ensure column order matches sample submission
    # Sample: image_id, healthy, multiple_diseases, rust, scab
    # Our Config.CLASS_LABELS are sorted alphabetically: healthy, multiple_diseases, rust, scab
    # So the order is already correct.

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


if __name__ == "__main__":
    main()
