import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library import config, utils, preprocessing, dataset, model


def predict(
    model_dir=config.WORKING_DIR,
    output_path=config.SUBMISSION_PATH,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    device=config.DEVICE,
):
    """
    Runs inference on the test set using the ensemble of trained fold models.

    Args:
        model_dir (str): Directory containing the trained model checkpoints.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        device (str): Computation device ('cpu' or 'cuda').
    """
    # Initialize Logger
    logger = utils.get_logger("Inference")
    logger.info("Initializing Inference Pipeline...")

    # Ensure reproducibility
    utils.seed_everything(config.SEED)

    # 1. Load Data
    # We use the library function which handles caching and loading from metadata.
    # prepare_datasets returns ((train), (val), (test))
    logger.info("Loading test dataset...")
    _, _, test_data = preprocessing.prepare_datasets(load_cached_data=True)
    test_images, test_ids = test_data

    logger.info(f"Test Data Shape: {test_images.shape}")

    # 2. Create Dataset and DataLoader
    test_dataset = dataset.MGMTDataset(
        test_images, test_ids, transform=dataset.get_transforms("test"), is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    # 3. Ensemble Inference
    num_folds = config.NUM_FOLDS
    # Array to store sum of probabilities for averaging
    ensemble_probs = np.zeros((len(test_ids), 1), dtype=np.float32)
    valid_models_count = 0

    logger.info(f"Starting inference with {num_folds}-fold ensemble...")

    for fold in range(num_folds):
        model_path = os.path.join(model_dir, f"best_model_fold{fold}.pth")

        if not os.path.exists(model_path):
            logger.warning(
                f"Checkpoint for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        logger.info(f"Processing Fold {fold}...")

        # Initialize Model
        # pretrained=False because we are loading custom weights
        net = model.MGMTNet(pretrained=False)
        net.to(device)

        # Load Weights
        try:
            state_dict = torch.load(model_path, map_location=device)
            net.load_state_dict(state_dict)
        except Exception as e:
            logger.error(f"Failed to load model fold {fold}: {e}")
            continue

        net.eval()

        fold_probs = []

        # Inference Loop
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)

                # Forward pass
                logits = net(images)

                # Apply Sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                fold_probs.append(probs.cpu().numpy())

        # Concatenate batches
        fold_probs = np.concatenate(fold_probs)

        # Accumulate
        ensemble_probs += fold_probs
        valid_models_count += 1

        # Cleanup to save memory
        del net
        torch.cuda.empty_cache()

    # 4. Aggregate Results
    if valid_models_count > 0:
        avg_probs = ensemble_probs / valid_models_count
        logger.info(
            f"Inference complete. Averaged predictions from {valid_models_count} models."
        )
    else:
        logger.error("No valid models found! Defaulting predictions to 0.5.")
        avg_probs = np.full((len(test_ids), 1), 0.5)

    # 5. Generate Submission File
    df_sub = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": avg_probs.flatten()})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df_sub.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Submission Preview:\n{df_sub.head()}")
