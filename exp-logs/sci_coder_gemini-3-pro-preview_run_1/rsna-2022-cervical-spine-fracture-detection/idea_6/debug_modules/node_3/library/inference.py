import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from library import config, models, generate_features


def load_stage3_model(device):
    """
    Loads the Stage 3 Anatomical Bi-GRU model.
    """
    model = models.AnatomicalBiGRU()
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "fracture_aggregator.pth")

    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded Stage 3 checkpoint: {checkpoint_path}")
        except Exception as e:
            print(
                f"Failed to load Stage 3 checkpoint: {e}. Using random initialization."
            )
    else:
        print("Stage 3 checkpoint not found. Using random initialization.")

    model.to(device)
    model.eval()
    return model


def predict_study(study_uid, features, stage3_model, device):
    """
    Runs Stage 3 inference on pre-computed features for a single study.

    Args:
        study_uid (str): Study Instance UID.
        features (np.ndarray): Array of shape (Seq_Len, 1287) containing [Visual(1280) + Anat(7)].
        stage3_model (nn.Module): The Bi-GRU model.
        device (str): Compute device.

    Returns:
        dict: Dictionary mapping target names to probabilities.
    """
    targets = config.VERTEBRAE_CLASSES + ["patient_overall"]

    # Handle empty sequence (no images or processing failure)
    if features.shape[0] == 0:
        # Return low probability defaults
        return {t: 0.01 for t in targets}

    # Prepare inputs
    # Features: (Seq, 1287) -> Split into Visual (1280) and Anat (7)
    visual_dim = config.STAGE2_CONFIG["feature_dim"]

    visual_feats_np = features[:, :visual_dim]
    anat_ids_np = features[:, visual_dim:]

    # Convert to tensors and add batch dimension
    visual_tensor = (
        torch.from_numpy(visual_feats_np).float().unsqueeze(0).to(device)
    )  # (1, Seq, 1280)
    anat_tensor = (
        torch.from_numpy(anat_ids_np).float().unsqueeze(0).to(device)
    )  # (1, Seq, 7)

    with torch.no_grad():
        # Forward pass
        logits = stage3_model(visual_tensor, anat_tensor)  # (1, 8)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # (8,)

    return dict(zip(targets, probs))


def run_inference(load_cached_data=False):
    """
    Main inference driver. Generates submission.csv.
    """
    print("Starting Inference Pipeline...")

    # 1. Setup
    device = config.DEVICE
    cache_dir = os.path.join(config.WORKING_DIR, "cache", "test_features_inference")
    os.makedirs(cache_dir, exist_ok=True)

    # 2. Load Metadata
    # We use test_metadata.csv which contains unique studies
    if not os.path.exists(config.TEST_METADATA_PATH):
        print(
            f"Test metadata not found at {config.TEST_METADATA_PATH}. Cannot proceed."
        )
        return

    test_df = pd.read_csv(config.TEST_METADATA_PATH)
    study_uids = test_df["StudyInstanceUID"].unique()
    image_paths = dict(zip(test_df["StudyInstanceUID"], test_df["image_path"]))

    print(f"Found {len(study_uids)} studies in test set.")

    # 3. Load Models
    # Load Stage 1 & 2 (using helper from generate_features)
    unet, encoder = generate_features.load_models(device)
    # Load Stage 3
    aggregator = load_stage3_model(device)

    # 4. Processing Loop
    submission_rows = []

    for uid in study_uids:
        feature_path = os.path.join(cache_dir, f"{uid}.npy")
        features = None

        # A. Feature Retrieval (Cache or Compute)
        if load_cached_data and os.path.exists(feature_path):
            try:
                features = np.load(feature_path)
            except Exception as e:
                print(f"Error loading cached features for {uid}: {e}. Re-computing.")

        if features is None:
            # Compute features using Stage 1 + Stage 2
            img_path = image_paths.get(uid, "")
            if not img_path:
                # Should not happen if metadata is correct
                features = np.zeros((0, 1287), dtype=np.float32)
            else:
                try:
                    features = generate_features.process_study(
                        uid, img_path, unet, encoder, device
                    )
                except Exception as e:
                    print(f"Error processing study {uid}: {e}")
                    features = np.zeros((0, 1287), dtype=np.float32)

            # Save to cache
            np.save(feature_path, features)

        # B. Prediction (Stage 3)
        preds = predict_study(uid, features, aggregator, device)

        # C. Format Rows
        # Format: [StudyUID]_[Target], [Probability]
        for target, prob in preds.items():
            row_id = f"{uid}_{target}"
            submission_rows.append({"row_id": row_id, "fractured": prob})

    # 5. Save Submission
    submission_df = pd.DataFrame(submission_rows)

    # Ensure column order
    submission_df = submission_df[["row_id", "fractured"]]

    save_path = config.SUBMISSION_PATH
    submission_df.to_csv(save_path, index=False)

    print(f"Inference completed. Submission saved to {save_path}")
    print(f"Total rows generated: {len(submission_df)}")
    if len(submission_df) > 0:
        print("Sample predictions:")
        print(submission_df.head())
