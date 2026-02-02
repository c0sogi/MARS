import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import WhaleDataset, get_transforms
from library.utils import seed_everything


def generate_pseudo_labels(models, device, label_encoder, load_cached_data=True):
    """
    Generates pseudo-labels for the unlabeled test set using an ensemble of trained models.
    Filters predictions based on confidence threshold and excludes 'new_whale'.
    Merges high-confidence pseudo-labels with the original training data.

    Args:
        models (list of nn.Module): List of trained models for the ensemble.
        device (str or torch.device): Device to perform inference on.
        label_encoder (LabelEncoder): Fitted label encoder to decode predictions.
        load_cached_data (bool): Whether to load previously generated pseudo-labels from disk.

    Returns:
        pd.DataFrame: A DataFrame containing the original training data plus the new pseudo-labels.
    """
    # Define cache path
    cache_path = os.path.join(Config.WORKING_DIR, "pseudo_labeled_train.csv")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached pseudo-labeled data from {cache_path}")
        return pd.read_csv(cache_path)

    print("Cache not found or ignored. Generating pseudo-labels from scratch...")

    # 2. Prepare Test Data
    df_test = pd.read_csv(Config.TEST_CSV)

    # Handle Debug Mode
    if Config.DEBUG:
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"Debug mode: processing {len(df_test)} test samples.")

    # Create Dataset and Loader
    # We use 'test' transforms (Resize + Normalize)
    test_dataset = WhaleDataset(
        df_test, transform=get_transforms("test"), label_encoder=None, is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.get_batch_size(inference=True),
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    # 3. Inference Loop
    # Ensure models are in eval mode
    for model in models:
        model.eval()
        model.to(device)

    pseudo_samples = []

    print(f"Starting inference on {len(df_test)} images using {len(models)} models...")

    with torch.no_grad():
        for images, image_names in test_loader:
            images = images.to(device)

            # Accumulate logits from all models
            ensemble_logits = None

            for model in models:
                # TTA: Original Image
                # Note: ArcFace model returns scaled cosine similarity when labels=None
                logits_orig = model(images)

                # TTA: Horizontal Flip
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)

                # Average views for this model
                model_avg_logits = (logits_orig + logits_flip) / 2.0

                if ensemble_logits is None:
                    ensemble_logits = model_avg_logits
                else:
                    ensemble_logits += model_avg_logits

            # Average across ensemble
            ensemble_logits /= len(models)

            # Apply Softmax to get probabilities
            # ArcFace logits are scaled (s=30), so softmax produces sharp probabilities
            probs = F.softmax(ensemble_logits, dim=1)

            # Get max probability and corresponding class index
            max_probs, indices = torch.max(probs, dim=1)

            # Move to CPU
            max_probs = max_probs.cpu().numpy()
            indices = indices.cpu().numpy()

            # Filter and Collect
            for i in range(len(image_names)):
                prob = max_probs[i]
                idx = indices[i]
                img_name = image_names[i]

                # Check Confidence Threshold
                if prob > Config.PSEUDO_LABEL_THRESHOLD:
                    pred_label = label_encoder.inverse_transform([idx])[0]

                    # Exclude 'new_whale'
                    # We only want to reinforce known classes to help with few-shot learning
                    if pred_label != "new_whale":
                        # Construct file path relative to input directory
                        # Based on metadata/test.csv, paths are "test/<filename>"
                        file_path = os.path.join("test", img_name)

                        pseudo_samples.append(
                            {
                                "Image": img_name,
                                "Id": pred_label,
                                "file_path": file_path,
                            }
                        )

    print(
        f"Inference complete. Found {len(pseudo_samples)} high-confidence pseudo-labels (Threshold: {Config.PSEUDO_LABEL_THRESHOLD})."
    )

    # 4. Merge with Original Training Data
    df_train = pd.read_csv(Config.TRAIN_CSV)

    if len(pseudo_samples) > 0:
        df_pseudo = pd.DataFrame(pseudo_samples)

        # Concatenate
        df_combined = pd.concat([df_train, df_pseudo], axis=0).reset_index(drop=True)
        print(
            f"Merged dataset size: {len(df_combined)} (Original: {len(df_train)} + Pseudo: {len(df_pseudo)})"
        )
    else:
        df_combined = df_train
        print("No pseudo-labels generated. Returning original training set.")

    # 5. Save to Cache
    # Ensure directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df_combined.to_csv(cache_path, index=False)
    print(f"Saved combined dataset to {cache_path}")

    return df_combined
