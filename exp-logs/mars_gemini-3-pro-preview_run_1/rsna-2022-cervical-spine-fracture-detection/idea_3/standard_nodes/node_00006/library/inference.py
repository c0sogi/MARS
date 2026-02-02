import os
import torch
import numpy as np
import pandas as pd
import cv2
import glob
from torch.utils.data import DataLoader

from library.config import Config
from library.models import SpineLocalizer, SliceEncoder, SequenceAggregator
from library.utils import process_dicom, crop_image


def load_models(device):
    """
    Loads all three stages of models with their trained weights.
    """
    # 1. Localizer
    localizer = SpineLocalizer(pretrained=False)
    loc_path = os.path.join(Config.CHECKPOINT_DIR, "spine_localizer.pth")
    if os.path.exists(loc_path):
        localizer.load_state_dict(torch.load(loc_path, map_location=device))
    localizer.to(device)
    localizer.eval()

    # 2. Encoder
    # Note: SliceEncoder expects backbone_name. We use the one from Config.
    encoder = SliceEncoder(backbone_name=Config.ENCODER_BACKBONE, pretrained=False)
    enc_path = os.path.join(Config.CHECKPOINT_DIR, "slice_encoder.pth")
    if os.path.exists(enc_path):
        encoder.load_state_dict(torch.load(enc_path, map_location=device))
    encoder.to(device)
    encoder.eval()

    # 3. Aggregator
    # We need to know input_dim. ResNet50 usually outputs 2048.
    # EfficientNetV2-S outputs 1280.
    # We instantiate a dummy encoder to check or hardcode based on config.
    # Based on library.models, SliceEncoder uses timm or torchvision.
    # We'll infer dimension dynamically or assume ResNet50 (2048).
    dummy_encoder = SliceEncoder(
        backbone_name=Config.ENCODER_BACKBONE, pretrained=False
    )
    input_dim = dummy_encoder.out_dim

    aggregator = SequenceAggregator(
        input_dim=input_dim,
        hidden_dim=Config.RNN_HIDDEN_SIZE,
        num_layers=Config.RNN_NUM_LAYERS,
        dropout=Config.RNN_DROPOUT,
    )
    agg_path = os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth")
    if os.path.exists(agg_path):
        aggregator.load_state_dict(torch.load(agg_path, map_location=device))
    aggregator.to(device)
    aggregator.eval()

    return localizer, encoder, aggregator


def get_spine_centers(images, model, device):
    """
    Predicts spine centers for a list of images.
    """
    input_size = Config.LOCALIZER_IMG_SIZE
    batch_size = Config.LOCALIZER_BATCH_SIZE
    centers = []

    # Process in batches
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        batch_tensors = []

        for img in batch_imgs:
            # Resize
            resized = cv2.resize(img, (input_size[1], input_size[0]))
            # To Tensor (1, H, W)
            t = torch.from_numpy(resized).unsqueeze(0).float()
            batch_tensors.append(t)

        batch_input = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            logits = model(batch_input)
            masks = torch.sigmoid(logits)
            masks = (masks > 0.5).float().cpu().numpy()

        for j, mask in enumerate(masks):
            mask = mask[0]  # (H, W)
            orig_idx = i + j
            orig_h, orig_w = images[orig_idx].shape

            if np.sum(mask) > 0:
                indices = np.argwhere(mask)
                y_center = np.mean(indices[:, 0])
                x_center = np.mean(indices[:, 1])

                # Scale back
                scale_y = orig_h / input_size[0]
                scale_x = orig_w / input_size[1]

                centers.append((y_center * scale_y, x_center * scale_x))
            else:
                # Default to center
                centers.append((orig_h // 2, orig_w // 2))

    return centers


def predict_study(study_uid, image_dir, models, device):
    """
    Runs the full pipeline for a single study.
    """
    localizer, encoder, aggregator = models

    # 1. Load Images
    dcm_files = glob.glob(os.path.join(image_dir, "*.dcm"))
    if not dcm_files:
        # Return default probabilities if no images found
        return [0.5] * 8

    try:
        dcm_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except:
        dcm_files.sort()

    # Load and process all slices
    images = [process_dicom(f) for f in dcm_files]
    num_slices = len(images)

    # 2. Localize
    centers = get_spine_centers(images, localizer, device)

    # 3. Encode
    # Create 2.5D stacks and batch them
    batch_size = Config.ENCODER_BATCH_SIZE
    crop_h, crop_w = Config.ENCODER_CROP_SIZE
    study_features = []

    stack_buffer = []

    for i in range(num_slices):
        img_c = images[i]
        img_p = images[i - 1] if i > 0 else img_c
        img_n = images[i + 1] if i < num_slices - 1 else img_c

        cy, cx = centers[i]

        crop_c = crop_image(img_c, (cy, cx), (crop_h, crop_w))
        crop_p = crop_image(img_p, (cy, cx), (crop_h, crop_w))
        crop_n = crop_image(img_n, (cy, cx), (crop_h, crop_w))

        # Stack (3, H, W)
        stack = np.stack([crop_p, crop_c, crop_n], axis=0)
        stack_buffer.append(torch.from_numpy(stack).float())

        if len(stack_buffer) == batch_size or i == num_slices - 1:
            batch_tensor = torch.stack(stack_buffer).to(device)
            with torch.no_grad():
                feats = encoder(batch_tensor)
                study_features.append(feats)
            stack_buffer = []

    if not study_features:
        return [0.5] * 8

    # Concatenate all features: (Seq_Len, Dim)
    full_features = torch.cat(study_features, dim=0)

    # 4. Aggregate
    # Add batch dimension: (1, Seq_Len, Dim)
    input_seq = full_features.unsqueeze(0)

    with torch.no_grad():
        logits = aggregator(input_seq)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # (8,)

    return probs


def generate_predictions(debug=False):
    """
    Main inference function.
    Generates predictions for the test set and saves submission.csv.
    """
    Config.setup()
    device = Config.DEVICE
    print(f"Running Inference on device: {device}")

    # Load Metadata
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    if debug:
        test_df = test_df.head(5)

    # Load Models
    models = load_models(device)

    results = []
    target_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    print(f"Processing {len(test_df)} studies...")

    for idx, row in test_df.iterrows():
        uid = row["StudyInstanceUID"]
        rel_path = row["image_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Predict
        try:
            probs = predict_study(uid, full_path, models, device)
        except Exception as e:
            print(f"Error processing {uid}: {e}")
            probs = [0.05] * 8  # Conservative fallback

        # Format results
        for i, col in enumerate(target_cols):
            row_id = f"{uid}_{col}"
            results.append({"row_id": row_id, "fractured": float(probs[i])})

    # Save Submission
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())


if __name__ == "__main__":
    generate_predictions()
