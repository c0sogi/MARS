import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_test_loader
from library.models import WhaleModel
from library.loss import ArcFaceLoss


def load_ensemble(device, num_classes):
    """
    Loads the ensemble of models and their corresponding ArcFace loss modules
    (which contain the class centers).

    Args:
        device (str): Device to load models onto.
        num_classes (int): Number of classes to initialize the ArcFace head.

    Returns:
        tuple: (list of models, list of criterions)
    """
    models = []
    criterions = []

    print(f"Loading ensemble of {len(Config.ENSEMBLE_MODELS)} models...")

    for model_cfg in Config.ENSEMBLE_MODELS:
        arch = model_cfg["arch"]
        name = model_cfg["name"]
        # seed is used during training for initialization, not explicitly needed for loading
        # unless we were re-initializing, but we load state_dicts.

        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"{name}_best.pth")

        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found at {ckpt_path}. Skipping this model.")
            continue

        print(f"Loading {name} ({arch})...")

        # Initialize Model
        # pretrained=False because we are loading specific weights
        model = WhaleModel(arch=arch, num_classes=num_classes, pretrained=False)
        model.to(device)

        # Initialize Criterion
        # We need the criterion because it holds the 'weight' (class centers)
        # required to compute cosine similarity for the ArcFace metric.
        criterion = ArcFaceLoss(num_classes=num_classes)
        criterion.to(device)

        # Load Checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        criterion.load_state_dict(checkpoint["criterion_state_dict"])

        # Set to Eval mode
        model.eval()
        criterion.eval()

        models.append(model)
        criterions.append(criterion)

    return models, criterions


def predict_ensemble():
    """
    Main inference function.
    1. Loads test data.
    2. Loads ensemble models.
    3. Performs inference with TTA (Horizontal Flip).
    4. Aggregates predictions.
    5. Saves submission file.
    """
    # Ensure reproducibility
    seed_everything(42)
    device = Config.DEVICE

    # 1. Load Data
    # We load cached data to ensure we have the exact same class list as training
    test_loader, classes = get_test_loader(load_cached_data=True)
    num_classes = len(classes)

    # 2. Load Ensemble
    models, criterions = load_ensemble(device, num_classes)

    if not models:
        raise RuntimeError(
            "No models were successfully loaded. Cannot proceed with inference."
        )

    print("Starting inference...")

    results = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, image_names in test_loader:
            images = images.to(device)
            batch_size = images.size(0)

            # Accumulator for ensemble logits
            # Shape: [Batch Size, Num Classes]
            ensemble_logits = torch.zeros((batch_size, num_classes), device=device)

            # Iterate through each model in the ensemble
            for i in range(len(models)):
                model = models[i]
                criterion = criterions[i]

                # Retrieve normalized class centers from the ArcFace module
                # Shape: [Num Classes, Embedding Size]
                class_centers = F.normalize(criterion.weight, p=2, dim=1)

                # --- View 1: Original Image ---
                emb_orig = model(images)
                emb_orig_norm = F.normalize(emb_orig, p=2, dim=1)
                # Cosine Similarity = Dot product of normalized vectors
                logits_orig = F.linear(emb_orig_norm, class_centers)

                # --- View 2: Horizontally Flipped Image (TTA) ---
                if Config.TTA_FLIP:
                    # Flip along width (dim 3)
                    images_flip = torch.flip(images, dims=[3])

                    emb_flip = model(images_flip)
                    emb_flip_norm = F.normalize(emb_flip, p=2, dim=1)
                    logits_flip = F.linear(emb_flip_norm, class_centers)

                    # Average the logits for this model
                    model_logits = (logits_orig + logits_flip) / 2.0
                else:
                    model_logits = logits_orig

                # Add to ensemble accumulator
                ensemble_logits += model_logits

            # Average across all ensemble members
            ensemble_logits /= len(models)

            # Retrieve Top K Predictions
            # We need the indices of the highest cosine similarities
            _, top_indices = torch.topk(ensemble_logits, k=Config.TOP_K, dim=1)

            # Move to CPU for processing
            top_indices = top_indices.cpu().numpy()

            # Map indices back to class strings
            for idx, indices in enumerate(top_indices):
                img_name = image_names[idx]

                # Convert indices to string labels
                pred_labels = [classes[i] for i in indices]

                # Join with spaces as required by submission format
                pred_string = " ".join(pred_labels)

                results.append({"Image": img_name, "Id": pred_string})

    # 3. Generate Submission File
    df_sub = pd.DataFrame(results)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Inference complete. Submission saved to {Config.SUBMISSION_FILE}")
    print("Sample predictions:")
    print(df_sub.head())
