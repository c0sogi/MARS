import os
import torch
import pandas as pd
import numpy as np
from library import config, dataset, model


def generate_submission(checkpoint_path=None, debug_size=None):
    """
    Generates a submission file for the test dataset using a trained model.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to 'best_model.pth' in the working directory.
        debug_size (int, optional): Number of samples to process for debugging purposes.
                                    If None, processes the entire test set.
    """
    # 1. Setup Paths and Device
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Model
    # We initialize with pretrained=False since we are loading custom weights
    net = model.DeepSupervisedResNet50(pretrained=False)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net = net.to(device)
    net.eval()

    # 3. Load Data
    test_loader = dataset.get_test_loader()

    # 4. Load Mappings (Index -> Category ID)
    mappings = config.get_hierarchy_mappings()
    idx_to_cat = mappings["idx_to_cat"]

    # 5. Inference Loop
    predictions = []
    processed_samples = 0

    with torch.no_grad():
        for images, sample_ids in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = net(images)

            # We use the fine-grained head for the final prediction
            logits = outputs["fine"]
            _, preds = torch.max(logits, 1)

            # Move to CPU for processing
            preds_cpu = preds.cpu().numpy()
            ids_cpu = sample_ids.numpy()

            # Map predictions back to category IDs
            for pid, cls_idx in zip(ids_cpu, preds_cpu):
                cat_id = idx_to_cat[cls_idx]
                predictions.append({"_id": pid, "category_id": cat_id})

            processed_samples += images.size(0)

            # Debugging control
            if debug_size is not None and processed_samples >= debug_size:
                break

    # 6. Save Submission
    df_sub = pd.DataFrame(predictions)

    # Ensure correct column order
    df_sub = df_sub[["_id", "category_id"]]

    # Save to CSV
    # Ensure the directory exists (config usually handles this, but being safe)
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
