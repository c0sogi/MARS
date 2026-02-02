import os
import torch
import pandas as pd
import numpy as np
from library import config, data, model, utils


def predict_submission(
    test_metadata_path=config.TEST_METADATA_PATH,
    model_path=config.MODEL_SAVE_PATH,
    submission_output_path=config.SUBMISSION_FILE,
    batch_size=config.BATCH_SIZE,
    device=None,
):
    """
    Generates predictions for the test set using the trained model and Test-Time Augmentation (TTA).

    Args:
        test_metadata_path (str): Path to the test metadata CSV.
        model_path (str): Path to the trained model weights (.pth).
        submission_output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        device (torch.device, optional): Device to run inference on.
    """
    # Ensure reproducibility
    utils.set_seed(config.SEED)

    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # 1. Verify Paths
    if not os.path.exists(test_metadata_path):
        print(f"Error: Test metadata file not found at {test_metadata_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    # 2. Prepare Data
    # Load metadata manually first to retrieve IDs in the correct order
    df_test = pd.read_csv(test_metadata_path)
    test_ids = df_test["BraTS21ID"].values

    # Initialize DataLoader
    # shuffle=False is critical to ensure predictions align with test_ids
    test_loader = data.get_dataloader(
        metadata_path=test_metadata_path,
        batch_size=batch_size,
        is_train=False,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Load Model
    net = model.AsymmetricEfficientNet().to(device)

    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict)
        print(f"Successfully loaded model weights from {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    net.eval()

    # 4. Inference Loop with TTA
    all_predictions = []
    print(
        "Starting inference with Test-Time Augmentation (Original + HFlip + VFlip)..."
    )

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)

            # TTA 1: Original Input
            outputs_orig = net(inputs)
            probs_orig = torch.sigmoid(outputs_orig)

            # TTA 2: Horizontal Flip (Flip Width - dim 3)
            # Input shape is (Batch, Channels, Height, Width)
            inputs_h = torch.flip(inputs, dims=[3])
            outputs_h = net(inputs_h)
            probs_h = torch.sigmoid(outputs_h)

            # TTA 3: Vertical Flip (Flip Height - dim 2)
            inputs_v = torch.flip(inputs, dims=[2])
            outputs_v = net(inputs_v)
            probs_v = torch.sigmoid(outputs_v)

            # Average Predictions
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            # Flatten and store
            batch_preds = avg_probs.cpu().numpy().flatten()
            all_predictions.extend(batch_preds)

    # 5. Save Submission
    # Verify length consistency
    if len(all_predictions) != len(test_ids):
        print(
            f"Warning: Number of predictions ({len(all_predictions)}) does not match number of test IDs ({len(test_ids)})."
        )

    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": all_predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(submission_output_path), exist_ok=True)

    submission_df.to_csv(submission_output_path, index=False)
    print(f"Submission saved to {submission_output_path}")
