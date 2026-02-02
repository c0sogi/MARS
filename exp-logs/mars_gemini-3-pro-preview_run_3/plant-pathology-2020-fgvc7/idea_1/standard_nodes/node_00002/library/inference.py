import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library import utils, data, model


def predict_and_submit(debug: bool = Config.DEBUG):
    """
    Loads the best trained model, performs inference on the test set,
    and generates the submission CSV file.

    Args:
        debug (bool): If True, runs inference on a subset of the test data
                      (controlled by the data loader logic).
    """
    # 1. Setup Device
    device = utils.get_device()
    print(f"Running inference on {device}...")

    # 2. Initialize Model Architecture
    # We initialize with pretrained=False because we are about to load specific weights.
    # This avoids unnecessary downloads of ImageNet weights during the inference phase.
    net = model.AppleDiseaseModel(
        model_name=Config.MODEL_NAME,
        pretrained=False,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    # 3. Load Best Model Weights
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if os.path.exists(best_model_path):
        print(f"Loading model weights from {best_model_path}")
        state_dict = torch.load(best_model_path, map_location=device)
        net.load_state_dict(state_dict)
    else:
        print(f"Warning: {best_model_path} not found. Using random initialization.")

    net.eval()

    # 4. Get Test DataLoader
    # get_dataloaders returns (train, val, test). We only need the test loader.
    _, _, test_loader = data.get_dataloaders(debug=debug)

    # 5. Inference Loop
    results = []
    print("Starting inference...")

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            # TTA: 1. Original
            logits_orig = net(images)
            probs_orig = torch.softmax(logits_orig, dim=1)

            # TTA: 2. Horizontal Flip
            images_hflip = torch.flip(images, [3])
            logits_hflip = net(images_hflip)
            probs_hflip = torch.softmax(logits_hflip, dim=1)

            # TTA: 3. Vertical Flip
            images_vflip = torch.flip(images, [2])
            logits_vflip = net(images_vflip)
            probs_vflip = torch.softmax(logits_vflip, dim=1)

            # Average probabilities
            probs = (probs_orig + probs_hflip + probs_vflip) / 3.0
            probs = probs.cpu().numpy()

            # Map predictions to image_ids
            for img_id, prob_vector in zip(image_ids, probs):
                row = {"image_id": img_id}
                # Map each probability to its corresponding class name
                for idx, class_name in enumerate(Config.CLASSES):
                    row[class_name] = prob_vector[idx]
                results.append(row)

    # 6. Create Submission DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure correct column order: image_id, healthy, multiple_diseases, rust, scab
    cols = ["image_id"] + Config.CLASSES
    submission_df = submission_df[cols]

    # 7. Save to CSV
    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Optional: Print first few rows to verify format
    print(submission_df.head())
