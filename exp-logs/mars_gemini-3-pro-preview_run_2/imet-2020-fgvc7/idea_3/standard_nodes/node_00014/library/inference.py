import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything, optimize_threshold
from library.dataset import get_dataloaders
from library.models import ArtworkClassifier, validate, inference
from library.loss import AsymmetricLoss


def predict_with_tta(model, loader, device):
    """
    Generates predictions for the test set using Test-Time Augmentation.
    Wraps the library.models.inference function which handles TTA logic (horizontal flip).

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Test data loader.
        device (torch.device): Computation device.

    Returns:
        tuple: (probabilities, image_ids)
    """
    # Force use_tta=True as per requirement
    return inference(model, loader, device, use_tta=True)


def run_inference(load_cached_data=True, debug=Config.debug):
    """
    Orchestrates the inference process for the Heterogeneous Ensemble.

    Steps:
    1. Setup environment and data loaders.
    2. For each model in the ensemble:
       - Check cache for existing predictions.
       - If not cached, load model, predict on Val (for threshold) and Test (for submission).
       - Cache the results.
    3. Aggregate predictions (mean) across the ensemble.
    4. Optimize the decision threshold using Validation data.
    5. Apply threshold to Test data and generate submission CSV.

    Args:
        load_cached_data (bool): If True, attempts to load predictions from ./working/idea_3/.
        debug (bool): If True, uses a subset of data for faster execution.
    """
    seed_everything(Config.seed)
    device = torch.device(Config.device)

    # Ensure working directory exists for caching
    os.makedirs(Config.working_dir, exist_ok=True)

    print(
        f"Starting Inference. Device: {device}, Debug: {debug}, Caching: {load_cached_data}"
    )

    # Get DataLoaders
    # We need val_loader for threshold optimization and test_loader for submission
    # train_loader is ignored here
    _, val_loader, test_loader = get_dataloaders(
        batch_size=Config.batch_size, num_workers=Config.num_workers, debug=debug
    )

    ensemble_val_probs = []
    ensemble_test_probs = []
    val_targets = None
    test_ids = None

    # Loss function needed for validate() signature, though we only use the probs/targets output
    criterion = AsymmetricLoss()

    for model_name in Config.model_names:
        print(f"\n--- Processing Model: {model_name} ---")

        # Define cache paths
        # We append '_debug' to filename if in debug mode to avoid mixing cache
        suffix = "_debug" if debug else ""
        val_cache_path = os.path.join(
            Config.working_dir, f"val_probs_{model_name}{suffix}.npy"
        )
        val_targets_cache_path = os.path.join(
            Config.working_dir, f"val_targets_{model_name}{suffix}.npy"
        )
        test_cache_path = os.path.join(
            Config.working_dir, f"test_probs_{model_name}{suffix}.npy"
        )
        test_ids_cache_path = os.path.join(
            Config.working_dir, f"test_ids_{model_name}{suffix}.npy"
        )

        # Try loading from cache
        loaded_from_cache = False
        if load_cached_data:
            if (
                os.path.exists(val_cache_path)
                and os.path.exists(val_targets_cache_path)
                and os.path.exists(test_cache_path)
                and os.path.exists(test_ids_cache_path)
            ):

                print(f"Loading cached predictions for {model_name}...")
                try:
                    model_val_probs = np.load(val_cache_path)
                    model_val_targets = np.load(val_targets_cache_path)
                    model_test_probs = np.load(test_cache_path)
                    model_test_ids = np.load(test_ids_cache_path, allow_pickle=True)
                    loaded_from_cache = True
                except Exception as e:
                    print(f"Failed to load cache: {e}. Re-running inference.")
            else:
                print(f"Cache not found for {model_name}. Running inference...")

        if not loaded_from_cache:
            # Load Model Weights
            weight_path = os.path.join(Config.working_dir, f"{model_name}_best.pth")
            if not os.path.exists(weight_path):
                print(
                    f"Error: Weights not found at {weight_path}. Skipping this model."
                )
                continue

            print(f"Loading weights from {weight_path}...")
            model = ArtworkClassifier(model_name, Config.num_classes).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))

            # Validation Inference (for thresholding)
            print("Running Validation Inference...")
            _, _, model_val_probs, model_val_targets = validate(
                model, val_loader, criterion, device
            )

            # Test Inference (with TTA)
            print("Running Test Inference with TTA...")
            model_test_probs, model_test_ids = predict_with_tta(
                model, test_loader, device
            )

            # Save to cache
            print("Caching results...")
            np.save(val_cache_path, model_val_probs)
            np.save(val_targets_cache_path, model_val_targets)
            np.save(test_cache_path, model_test_probs)
            np.save(test_ids_cache_path, model_test_ids)

            # Cleanup to free memory
            del model
            torch.cuda.empty_cache()

        # Collect results for aggregation
        ensemble_val_probs.append(model_val_probs)
        ensemble_test_probs.append(model_test_probs)

        # Keep track of targets/ids (consistency check: these should be identical across models)
        val_targets = model_val_targets
        test_ids = model_test_ids

    if not ensemble_val_probs:
        print("No predictions available from any model. Exiting.")
        return

    # --- Ensemble Aggregation ---
    print("\n--- Aggregating Ensemble Predictions ---")

    # Average probabilities across the ensemble
    avg_val_probs = np.mean(ensemble_val_probs, axis=0)
    avg_test_probs = np.mean(ensemble_test_probs, axis=0)

    # --- Threshold Optimization ---
    print("Optimizing Threshold on Validation Set...")
    best_thresh, best_score = optimize_threshold(avg_val_probs, val_targets)
    print(f"Optimal Threshold: {best_thresh}")
    print(f"Best Ensemble Val F1: {best_score}")

    # --- Submission Generation ---
    print("Generating Submission...")

    # Apply optimal threshold to test probabilities
    test_preds_bin = (avg_test_probs >= best_thresh).astype(int)

    submission_rows = []
    for i, img_id in enumerate(test_ids):
        # Get indices of active classes (where prediction is 1)
        pred_indices = np.where(test_preds_bin[i] == 1)[0]
        # Convert to space-separated string
        pred_str = " ".join(map(str, pred_indices))
        submission_rows.append({"id": img_id, "attribute_ids": pred_str})

    df_sub = pd.DataFrame(submission_rows)

    # Ensure submission directory exists
    os.makedirs(Config.submission_dir, exist_ok=True)
    df_sub.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
