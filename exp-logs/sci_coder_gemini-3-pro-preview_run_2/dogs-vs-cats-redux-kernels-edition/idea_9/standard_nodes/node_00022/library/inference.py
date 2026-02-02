import os
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.dataset import PetDataset, get_transforms
from library.models import get_model
from library.engine import predict


def predict_test_set():
    """
    Generates predictions for the test set using the trained heterogeneous ensemble.
    Loads models for all architectures and folds, performs inference with TTA,
    averages the results, and saves the submission file.
    """
    seed_everything(Config.SEED)

    print("Loading test metadata...")
    test_df = pd.read_csv(Config.TEST_META)

    # Handle Debug mode
    if Config.DEBUG:
        print(
            f"DEBUG mode enabled. Truncating test set to {Config.DEBUG_SUBSET_SIZE} samples."
        )
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

    # Create Dataset and DataLoader
    test_dataset = PetDataset(
        test_df, transforms=get_transforms(data_type="test"), mode="test"
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Dictionary to aggregate probabilities: {image_id: accumulated_probability}
    ensemble_probs = {}
    models_loaded_count = 0

    # Iterate over all defined architectures and folds
    for model_name in Config.MODEL_ARCHS:
        for fold in range(Config.N_FOLDS):
            # Construct checkpoint path (looking for the 'best' saved model)
            checkpoint_filename = f"best_{model_name}_fold_{fold}.pth"
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, checkpoint_filename)

            if not os.path.exists(checkpoint_path):
                print(
                    f"Warning: Checkpoint not found at {checkpoint_path}. Skipping this model."
                )
                continue

            print(f"Processing Model: {model_name} | Fold: {fold}")

            # Initialize model architecture
            # pretrained=False because we are loading our own trained weights
            model = get_model(model_name, pretrained=False, num_classes=1)
            model = model.to(Config.DEVICE)

            # Load weights
            load_checkpoint(checkpoint_path, model, device=Config.DEVICE)

            # Generate predictions (handles TTA internally if Config.TTA_FLIP is True)
            ids, probs = predict(model, test_loader, Config.DEVICE)

            # Aggregate predictions
            for img_id, prob in zip(ids, probs):
                if img_id not in ensemble_probs:
                    ensemble_probs[img_id] = 0.0
                ensemble_probs[img_id] += prob

            models_loaded_count += 1

            # Clean up memory
            del model
            torch.cuda.empty_cache()

    if models_loaded_count == 0:
        raise RuntimeError("No valid checkpoints found. Cannot generate submission.")

    print(
        f"\nEnsemble complete. Aggregated predictions from {models_loaded_count} models."
    )

    # Calculate average and prepare DataFrame
    submission_data = []
    for img_id, total_prob in ensemble_probs.items():
        avg_prob = total_prob / models_loaded_count
        submission_data.append({"id": int(img_id), "label": avg_prob})

    submission_df = pd.DataFrame(submission_data)

    # Sort by ID as per submission format requirements
    submission_df = submission_df.sort_values("id")

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
