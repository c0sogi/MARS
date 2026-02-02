import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import DogCatDataset, get_transforms
from library.models import get_model
from library.utils import load_checkpoint
from library.engine import predict, generate_submission


def predict_ensemble(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    device=Config.DEVICE,
    debug=Config.DEBUG,
    debug_samples=Config.DEBUG_SAMPLES,
):
    """
    Performs ensemble inference on the test set using models defined in Config.
    Generates a submission file with averaged probabilities.

    Args:
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of worker threads for data loading.
        device (str): Computation device ('cpu' or 'cuda').
        debug (bool): If True, runs on a subset of the data.
        debug_samples (int): Number of samples to use in debug mode.
    """
    # Temporarily override Config debug settings to control dataset size
    # This is necessary because DogCatDataset reads directly from Config
    original_debug = Config.DEBUG
    original_samples = Config.DEBUG_SAMPLES
    Config.DEBUG = debug
    Config.DEBUG_SAMPLES = debug_samples

    try:
        # 1. Prepare Data
        # Use 'test' transforms which resize without cropping
        test_dataset = DogCatDataset(split="test", transform=get_transforms("test"))

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )

        print(f"Starting ensemble inference on {len(test_dataset)} images...")

        # Accumulators for ensemble averaging
        accumulated_probs = None
        stored_ids = None
        models_run = 0

        # 2. Iterate through defined architectures
        for model_name in Config.MODEL_ARCHS:
            print(f"Processing model: {model_name}")

            # Construct checkpoint path
            # Assumes training script saves to: working/idea_3/{model_name}/model_best.pth
            checkpoint_path = os.path.join(
                Config.WORKING_DIR, model_name, "model_best.pth"
            )

            if not os.path.exists(checkpoint_path):
                print(f"  -> Checkpoint not found at {checkpoint_path}. Skipping.")
                continue

            # Initialize Model
            # pretrained=False because we are loading a full state dict from checkpoint
            try:
                model = get_model(model_name, pretrained=False)
                model.to(device)
            except Exception as e:
                print(f"  -> Failed to initialize model {model_name}: {e}")
                continue

            # Load Checkpoint
            try:
                load_checkpoint(checkpoint_path, model, device=device)
            except Exception as e:
                print(f"  -> Failed to load checkpoint for {model_name}: {e}")
                continue

            # Run Prediction
            # engine.predict handles TTA (horizontal flip) if Config.TTA_FLIP is True
            ids, probs = predict(model, test_loader, device, use_tta=Config.TTA_FLIP)

            # Convert to numpy for vector operations
            ids = np.array(ids)
            probs = np.array(probs)

            # Accumulate results
            if accumulated_probs is None:
                accumulated_probs = probs
                stored_ids = ids
            else:
                # Verify that IDs align perfectly between models
                if not np.array_equal(stored_ids, ids):
                    raise ValueError(
                        f"ID mismatch between models. Inference cannot proceed."
                    )
                accumulated_probs += probs

            models_run += 1
            print(f"  -> Model {model_name} inference complete.")

        # 3. Finalize and Save
        if models_run == 0:
            print("Error: No models were successfully run. Cannot generate submission.")
            return

        # Compute Arithmetic Mean
        avg_probs = accumulated_probs / models_run

        # Generate Submission CSV
        generate_submission(stored_ids, avg_probs, Config.SUBMISSION_PATH)
        print(f"Ensemble inference complete. Combined {models_run} models.")

    finally:
        # Restore original Config state
        Config.DEBUG = original_debug
        Config.DEBUG_SAMPLES = original_samples
