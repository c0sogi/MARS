import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.models import get_model
from library.dataset import get_datasets
from library.engine import predict


def run_inference(debug=Config.DEBUG, debug_sample_size=Config.DEBUG_SAMPLE_SIZE):
    """
    Runs the inference pipeline:
    1. Loads the test dataset.
    2. Iterates over all defined models and folds.
    3. Loads checkpoints and generates predictions with TTA.
    4. Aggregates predictions (ensemble averaging).
    5. Saves the submission file.

    Args:
        debug (bool): Whether to run in debug mode (subset of data).
        debug_sample_size (int): Number of samples to use in debug mode.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print("Initializing Inference...")
    print(f"Device: {device}")
    print(f"Models: {Config.MODELS}")
    print(f"Folds: {Config.N_FOLDS}")
    print(f"TTA Enabled: {Config.TTA_FLIP}")

    # 1. Get Test Data
    # We only need the test dataset here
    _, _, test_dataset = get_datasets(debug=debug, debug_sample_size=debug_sample_size)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
    )

    # Dictionary to store aggregated probabilities: {id: [prob_model_1, prob_model_2, ...]}
    ensemble_preds = {}

    # 2. Iterate over architectures and folds
    total_models_run = 0

    for model_name in Config.MODELS:
        for fold in range(Config.N_FOLDS):
            checkpoint_filename = f"{model_name}_fold_{fold}.pth"
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, checkpoint_filename)

            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path):
                print(
                    f"Warning: Checkpoint not found at {checkpoint_path}. Skipping this model."
                )
                continue

            print(f"Processing: {model_name} (Fold {fold})")

            # Load Model
            # num_classes=1 for binary classification
            model = get_model(model_name, pretrained=False, num_classes=1)

            try:
                state_dict = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_path}: {e}")
                continue

            model.to(device)

            # 3. Predict
            # library.engine.predict handles TTA (flip) if Config.TTA_FLIP is True
            # Returns list of dicts: [{'id': 1, 'label': 0.5}, ...]
            preds = predict(model, test_loader, device)

            # Aggregate results
            for item in preds:
                img_id = int(item["id"])
                prob = float(item["label"])

                if img_id not in ensemble_preds:
                    ensemble_preds[img_id] = []
                ensemble_preds[img_id].append(prob)

            total_models_run += 1

            # Clean up to save memory
            del model
            del state_dict
            torch.cuda.empty_cache()

    if total_models_run == 0:
        print("Error: No models were successfully loaded and run.")
        return

    print(
        f"Inference complete. Aggregating predictions from {total_models_run} models..."
    )

    # 4. Compute final average
    final_submission = []
    ids = sorted(ensemble_preds.keys())

    for img_id in ids:
        probs = ensemble_preds[img_id]
        avg_prob = np.mean(probs)
        final_submission.append({"id": img_id, "label": avg_prob})

    # 5. Create DataFrame and Save
    submission_df = pd.DataFrame(final_submission)

    # Ensure correct column order
    submission_df = submission_df[["id", "label"]]

    # Save to disk
    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print("Submission generated successfully.")
    print(submission_df.head())
