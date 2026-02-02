import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from library.config import Config
from library.models import AppleDiseaseModel
from library.data import get_loaders
from library.utils import seed_everything


def load_model_for_inference(model_name, checkpoint_path, device):
    """
    Instantiates the model architecture and loads weights from a checkpoint.

    Args:
        model_name (str): Name of the model architecture (timm).
        checkpoint_path (str): Path to the .pth checkpoint file.
        device (str): Device to load the model onto.

    Returns:
        model (nn.Module): The loaded model in eval mode, or None if loading fails.
    """
    # Instantiate model with pretrained=False since we are loading custom weights
    try:
        model = AppleDiseaseModel(model_name=model_name, pretrained=False)
    except Exception as e:
        print(f"Error instantiating model {model_name}: {e}")
        return None

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return None

    print(f"Loading weights for {model_name} from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint saving formats (full dict vs state_dict)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Load state dict
        msg = model.load_state_dict(state_dict, strict=True)
        print(f"Load status: {msg}")

        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load weights from {checkpoint_path}: {e}")
        return None


def predict_ensemble(models, loader, device, threshold=0.5, use_tta=True):
    """
    Runs inference on the loader using an ensemble of models.
    Implements TTA (Horizontal Flip) and NaN-safe probability aggregation.

    Args:
        models (list): List of loaded PyTorch models.
        loader (DataLoader): Test data loader.
        device (str): Device to run inference on.
        threshold (float): Threshold for binarizing probabilities.
        use_tta (bool): Whether to use Test Time Augmentation.

    Returns:
        pd.DataFrame: DataFrame with 'image' and 'labels' columns.
    """
    results = []
    total_batches = len(loader)
    print(
        f"Running inference (Models: {len(models)}, TTA={'Enabled' if use_tta else 'Disabled'})..."
    )

    # Ensure all models are in eval mode
    for m in models:
        m.eval()

    with torch.no_grad():
        for batch_idx, (images, _, image_ids) in enumerate(loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx + 1}/{total_batches}...", end="\r")

            images = images.to(device)
            batch_size = images.size(0)

            # Prepare inputs: [Original, Flipped] if TTA is enabled
            inputs = [images]
            if use_tta:
                inputs.append(torch.flip(images, dims=[3]))  # Flip width

            # Collect predictions: (NumModels, NumViews, BatchSize, NumClasses)
            # We will average views per model first, then aggregate models
            model_probabilities = []

            for model in models:
                view_preds = []
                for inp in inputs:
                    # Use mixed precision for inference speed/memory
                    with torch.cuda.amp.autocast(enabled=True):
                        logits = model(inp)
                        probs = torch.sigmoid(logits)
                    view_preds.append(probs)

                # Stack views: (NumViews, B, C)
                view_preds = torch.stack(view_preds)

                # Average TTA views for this model -> (B, C)
                avg_view_preds = torch.mean(view_preds, dim=0)
                model_probabilities.append(avg_view_preds)

            # Stack models: (NumModels, B, C)
            all_preds = torch.stack(model_probabilities)

            # NaN-Safe Aggregation
            final_batch_probs = []

            for i in range(batch_size):
                sample_preds = all_preds[:, i, :]  # (NumModels, C)

                valid_model_preds = []
                for m_idx in range(sample_preds.shape[0]):
                    p = sample_preds[m_idx]
                    # Check for NaN or Inf
                    if not (torch.isnan(p).any() or torch.isinf(p).any()):
                        valid_model_preds.append(p)

                if valid_model_preds:
                    # Average valid models
                    ensemble_p = torch.stack(valid_model_preds).mean(dim=0)
                else:
                    # Fallback: All models failed (extremely rare)
                    # Return zeros (no disease detected)
                    ensemble_p = torch.zeros(sample_preds.shape[1], device=device)

                final_batch_probs.append(ensemble_p)

            # Convert to numpy
            final_batch_probs = torch.stack(final_batch_probs).cpu().numpy()

            # Generate Labels
            for i, probs in enumerate(final_batch_probs):
                # Get class indices exceeding threshold
                pred_indices = np.where(probs > threshold)[0]

                # Map to class names
                pred_labels = [Config.CLASSES[idx] for idx in pred_indices]

                # Create space-delimited string
                label_str = " ".join(pred_labels)

                # If no labels predicted, default to 'healthy'
                if not label_str:
                    label_str = "healthy"

                results.append({"image": image_ids[i], "labels": label_str})

    print("\nInference complete.")
    return pd.DataFrame(results)


def generate_submission(checkpoint_list):
    """
    Main function to generate the submission file.

    Args:
        checkpoint_list (list): List of tuples (model_name, checkpoint_path).
                                Example: [(Config.MODEL_1_NAME, 'path/to/m1.pth'), ...]
    """
    seed_everything()
    device = Config.DEVICE

    # 1. Load Data
    # We only need the test loader
    print("Loading test data...")
    # Ensure cache directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    _, _, test_loader = get_loaders(load_cached_data=True)

    # 2. Load Models
    models = []
    if not checkpoint_list:
        print("No checkpoints provided. Cannot generate submission.")
        return

    for model_name, ckpt_path in checkpoint_list:
        model = load_model_for_inference(model_name, ckpt_path, device)
        if model is not None:
            models.append(model)

    if not models:
        print("No models were successfully loaded. Aborting submission generation.")
        return

    # 3. Predict
    df_submission = predict_ensemble(
        models=models,
        loader=test_loader,
        device=device,
        threshold=0.5,
        use_tta=Config.USE_TTA,
    )

    # 4. Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Head of submission:")
    print(df_submission.head())
