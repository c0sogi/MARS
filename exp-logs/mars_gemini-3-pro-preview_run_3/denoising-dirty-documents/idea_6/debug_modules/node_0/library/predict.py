import os
import torch
import numpy as np
import pandas as pd
from library import config
from library import model
from library import utils


def generate_predictions(
    model_path=config.MODEL_SAVE_PATH,
    metadata_path=config.TEST_METADATA_PATH,
    output_path=config.SUBMISSION_PATH,
    device=config.DEVICE,
):
    """
    Generates predictions for the test set using the trained RDN model.

    This function loads the trained model, iterates through the test images defined
    in the metadata, predicts the noise residual, reconstructs the clean image,
    and formats the results into a submission CSV.

    Args:
        model_path (str): Path to the saved model weights (.pth file).
        metadata_path (str): Path to the test metadata CSV file.
        output_path (str): Path where the submission CSV should be saved.
        device (str): Computation device ('cpu' or 'cuda').
    """
    # 1. Load Test Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Test metadata file not found at: {metadata_path}")

    df_test = pd.read_csv(metadata_path)
    test_ids = df_test["image_id"].tolist()
    input_paths = df_test["input_path"].tolist()

    print(f"Loaded test metadata. Found {len(test_ids)} images to process.")

    # 2. Initialize Model Architecture
    # We must use the exact same configuration as used during training
    rdn = model.RDN(
        channel=config.IMG_CHANNELS,
        growth_rate=config.RDN_GROWTH_RATE,
        num_features=config.RDN_NUM_FEATURES,
        num_blocks=config.RDN_NUM_BLOCKS,
        num_layers=config.RDN_LAYERS_PER_BLOCK,
        kernel_size=config.RDN_KERNEL_SIZE,
    ).to(device)

    # 3. Load Model Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    rdn.load_state_dict(state_dict)
    rdn.eval()

    # 4. Inference Loop
    predictions = []
    print("Starting inference...")

    with torch.no_grad():
        for i, rel_path in enumerate(input_paths):
            full_path = os.path.join(config.INPUT_DIR, rel_path)

            # Load image: returns (H, W) float32 array in [0, 1]
            try:
                img_noisy = utils.load_grayscale_image(full_path)
            except FileNotFoundError:
                print(f"Error: Image file missing at {full_path}. Skipping.")
                # In a real scenario, we might want to append a placeholder or fail.
                # Here we assume data integrity based on metadata checks.
                continue

            # Prepare input tensor: (1, 1, H, W)
            input_tensor = (
                torch.from_numpy(img_noisy).unsqueeze(0).unsqueeze(0).to(device)
            )

            # Predict Noise Residual
            noise_pred = rdn(input_tensor)

            # Reconstruct Clean Image
            # Formula: Clean = Noisy_Input - Predicted_Noise
            clean_pred = input_tensor - noise_pred

            # Post-processing
            # Remove batch and channel dimensions -> (H, W)
            clean_pred_np = clean_pred.squeeze().cpu().numpy()

            # Clip values to ensure valid pixel range [0, 1]
            clean_pred_np = np.clip(clean_pred_np, 0.0, 1.0)

            predictions.append(clean_pred_np)

            # Optional progress logging
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(input_paths)} images")

    # 5. Generate Submission File
    print(f"Generating submission file at {output_path}...")
    utils.format_submission(test_ids, predictions, output_path)
    print("Prediction process completed successfully.")
