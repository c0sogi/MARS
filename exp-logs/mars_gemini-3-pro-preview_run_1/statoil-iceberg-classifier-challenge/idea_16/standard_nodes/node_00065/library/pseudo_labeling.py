import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.data import load_dataset_data, IcebergDataset, get_transforms
from library.model import IcebergResNet
from library.utils import seed_everything


def generate_pseudo_labels(model_paths, load_cached_data=True, debug_limit=None):
    """
    Generates pseudo-labels from the test set using an ensemble of teacher models.
    Implements Exhaustive TTA (Original, HFlip, VFlip) and filters based on
    confidence and ensemble variance.

    Args:
        model_paths (list): List of file paths to the teacher model checkpoints.
        load_cached_data (bool): If True, attempts to load previously generated labels from disk.
        debug_limit (int, optional): Limit the number of test samples for debugging.

    Returns:
        tuple: (images, angles, labels) of the selected pseudo-labeled samples.
               Returns empty arrays if no samples satisfy the criteria.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, "pseudo_labels.npz")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading pseudo-labels from cache: {cache_path}")
        try:
            data = np.load(cache_path)
            return data["images"], data["angles"], data["labels"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    print("Starting Pseudo-Label Generation Process...")
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Load Test Data
    # load_dataset_data returns (images, angles, labels, ids)
    # For test, labels is None.
    test_images, test_angles, _, test_ids = load_dataset_data(
        "test", load_cached_data=True
    )

    if debug_limit is not None:
        print(f"Debug mode: Limiting test set to {debug_limit} samples.")
        test_images = test_images[:debug_limit]
        test_angles = test_angles[:debug_limit]
        test_ids = test_ids[:debug_limit]

    n_samples = len(test_images)
    print(f"Total Test Samples: {n_samples}")

    # 3. Prepare TTA Variants (Numpy)
    # Original: (N, 75, 75, 3)
    # Horizontal Flip: Flip width (axis 2)
    # Vertical Flip: Flip height (axis 1)
    tta_variants = {
        "orig": test_images,
        "hflip": np.flip(test_images, axis=2).copy(),
        "vflip": np.flip(test_images, axis=1).copy(),
    }

    # 4. Ensemble Inference
    # Shape: (N_samples, N_models)
    ensemble_preds = np.zeros((n_samples, len(model_paths)), dtype=np.float32)

    for m_idx, path in enumerate(model_paths):
        print(
            f"Running inference with Teacher Model {m_idx + 1}/{len(model_paths)}: {path}"
        )

        # Load Model
        model = IcebergResNet(pretrained=False)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        # Cite debug_lesson_12: Explicitly Enable Full Unpickling for Checkpoints with NumPy Data
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        # Support both full checkpoint dict and direct state_dict
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        # TTA Accumulator for this model
        model_tta_sum = np.zeros(n_samples, dtype=np.float32)

        for tta_name, tta_imgs in tta_variants.items():
            # Create Loader
            # Use 'test' transform (Resize -> Tensor)
            ds = IcebergDataset(
                tta_imgs,
                test_angles,
                ids=test_ids,
                transform=get_transforms(mode="test"),
            )

            loader = DataLoader(
                ds,
                batch_size=Config.BATCH_SIZE,  # Safe batch size
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Predict
            preds = []
            with torch.no_grad():
                for batch in loader:
                    imgs, angs = batch[0], batch[1]
                    imgs = imgs.to(device)
                    angs = angs.to(device)

                    outputs = model(imgs, angs)
                    probs = torch.sigmoid(outputs)
                    preds.extend(probs.cpu().numpy().flatten())

            model_tta_sum += np.array(preds)

        # Average TTA for this model
        ensemble_preds[:, m_idx] = model_tta_sum / len(tta_variants)

        # Clean up to save memory
        del model
        torch.cuda.empty_cache()

    # 5. Aggregate and Filter
    # Mean probability across ensemble
    mean_preds = np.mean(ensemble_preds, axis=1)
    # Standard deviation across ensemble (uncertainty)
    std_preds = np.std(ensemble_preds, axis=1)

    # Criteria 1: High Confidence
    # p > 0.95 OR p < 0.05
    conf_mask = (mean_preds > Config.SSL_CONFIDENCE_HIGH) | (
        mean_preds < Config.SSL_CONFIDENCE_LOW
    )

    # Criteria 2: Low Variance (Teacher Agreement)
    # std < 0.02
    var_mask = std_preds < Config.SSL_STD_THRESHOLD

    final_mask = conf_mask & var_mask
    selected_indices = np.where(final_mask)[0]

    n_selected = len(selected_indices)
    ratio = n_selected / n_samples if n_samples > 0 else 0
    print(f"Pseudo-Labeling Stats:")
    print(f"  Total Samples: {n_samples}")
    print(f"  Selected: {n_selected} ({ratio:.4%})")

    if n_selected == 0:
        print("  No samples met the strict criteria.")
        return np.array([]), np.array([]), np.array([])

    # 6. Extract and Format Data
    selected_images = test_images[selected_indices]
    selected_angles = test_angles[selected_indices]

    # Generate Labels: 1 if mean > 0.5 else 0
    # Since we filtered for > 0.95 or < 0.05, this is safe.
    selected_labels = (mean_preds[selected_indices] > 0.5).astype(np.float32)

    # 7. Save to Cache
    print(f"Saving {n_selected} pseudo-labeled samples to {cache_path}")
    np.savez(
        cache_path,
        images=selected_images,
        angles=selected_angles,
        labels=selected_labels,
    )

    return selected_images, selected_angles, selected_labels
