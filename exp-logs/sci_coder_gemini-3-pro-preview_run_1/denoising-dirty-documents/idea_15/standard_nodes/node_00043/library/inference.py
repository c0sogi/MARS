import os
import cv2
import torch
import numpy as np
import pandas as pd
from library.config import GlobalConfig, StreamAConfig, StreamBConfig
from library.model import UNet
from library.utils import seed_everything, d4_transform, d4_inverse_transform
from library.dataset import get_dataloader


def load_models(device):
    """
    Loads the ensemble of 10 models (5 Stream A, 5 Stream B) from the working directory.
    """
    models = []

    # Load Stream A Models (Context Specialists)
    for seed in GlobalConfig.STREAM_A_SEEDS:
        path = os.path.join(
            GlobalConfig.WORKING_DIR, f"{StreamAConfig.NAME}_seed_{seed}.pth"
        )
        if not os.path.exists(path):
            print(f"Warning: Model file not found: {path}")
            continue

        model = UNet(
            depth=StreamAConfig.DEPTH,
            encoder_filters=StreamAConfig.ENCODER_FILTERS,
            bottleneck_filters=StreamAConfig.BOTTLENECK_FILTERS,
            bottleneck_depth=StreamAConfig.BOTTLENECK_DEPTH,
            in_channels=1,
            out_channels=1,
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)

    # Load Stream B Models (Diversity Specialists)
    for seed in GlobalConfig.STREAM_B_SEEDS:
        path = os.path.join(
            GlobalConfig.WORKING_DIR, f"{StreamBConfig.NAME}_seed_{seed}.pth"
        )
        if not os.path.exists(path):
            print(f"Warning: Model file not found: {path}")
            continue

        model = UNet(
            depth=StreamBConfig.DEPTH,
            encoder_filters=StreamBConfig.ENCODER_FILTERS,
            bottleneck_filters=StreamBConfig.BOTTLENECK_FILTERS,
            bottleneck_depth=StreamBConfig.BOTTLENECK_DEPTH,
            in_channels=1,
            out_channels=1,
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)

    print(f"Successfully loaded {len(models)} models.")
    return models


def get_original_shapes(load_cached_data=True):
    """
    Retrieves the original dimensions (H, W) of test images.
    Caches the result to a CSV file to avoid re-reading images.
    """
    cache_path = os.path.join(GlobalConfig.WORKING_DIR, "test_shapes.csv")

    # Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            # Convert to dictionary {id: (h, w)}
            shapes = {str(row["id"]): (row["h"], row["w"]) for _, row in df.iterrows()}
            return shapes
        except Exception as e:
            print(f"Error loading shape cache: {e}. Recomputing.")

    # Compute from scratch
    shapes = {}
    data_list = []
    test_csv = os.path.join(GlobalConfig.METADATA_DIR, "test.csv")

    if os.path.exists(test_csv):
        df_meta = pd.read_csv(test_csv)
        for _, row in df_meta.iterrows():
            img_id = str(row["id"])
            path = os.path.join(GlobalConfig.INPUT_DIR, row["noisy_image_path"])

            # Read image to get shape
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                shapes[img_id] = (h, w)
                data_list.append({"id": img_id, "h": h, "w": w})

    # Save to cache
    if data_list:
        os.makedirs(GlobalConfig.WORKING_DIR, exist_ok=True)
        pd.DataFrame(data_list).to_csv(cache_path, index=False)

    return shapes


def generate_submission(load_cached_data=True, debug=False, epochs=None):
    """
    Generates the final submission file using the ensemble of models.

    Args:
        load_cached_data (bool): Whether to use cached metadata/shapes.
        debug (bool): If True, processes only a few images for testing.
        epochs (int): Unused, kept for signature compatibility.
    """
    seed_everything(GlobalConfig.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating submission on device: {device}")

    # 1. Load Models
    models = load_models(device)
    if not models:
        print("Error: No models available for inference.")
        return

    # 2. Get Original Shapes (to reverse padding)
    shapes = get_original_shapes(load_cached_data)

    # 3. Prepare DataLoader (Test mode)
    test_loader = get_dataloader(mode="test", shuffle=False)

    # 4. Initialize Submission File
    os.makedirs(GlobalConfig.SUBMISSION_DIR, exist_ok=True)
    sub_file = GlobalConfig.SUBMISSION_FILE

    # Write Header
    with open(sub_file, "w") as f:
        f.write("id,value\n")

    print(f"Starting inference on {len(test_loader)} images...")

    # 5. Inference Loop
    with torch.no_grad():
        for i, (noisy, img_id_batch) in enumerate(test_loader):
            if debug and i >= 2:
                print("Debug mode: stopping after 2 images.")
                break

            img_id = str(img_id_batch[0])

            # noisy is (1, 1, H_pad, W_pad)
            # Convert to numpy for D4 transforms (expects H, W)
            img_np = noisy.squeeze().cpu().numpy()

            # --- Test-Time Augmentation (TTA) ---
            # Group views by shape to handle rectangular images (Cite debug_lesson_2)
            views_by_shape = {}
            for k in range(GlobalConfig.TTA_VIEWS):
                view = d4_transform(img_np, k)
                shape = view.shape
                if shape not in views_by_shape:
                    views_by_shape[shape] = []
                views_by_shape[shape].append((k, view))

            # Run inference per shape group
            results = {}  # k -> prediction

            for shape, items in views_by_shape.items():
                ks = [x[0] for x in items]
                views = [x[1] for x in items]

                batch_np = np.stack(views)
                # (Batch, 1, H, W)
                batch_tensor = (
                    torch.from_numpy(batch_np).unsqueeze(1).to(device).float()
                )

                # Ensemble Prediction
                batch_accum = torch.zeros_like(batch_tensor)
                for model in models:
                    batch_accum += model(batch_tensor)

                batch_avg = batch_accum / len(models)
                preds_np = batch_avg.squeeze(1).cpu().numpy()

                for i, k in enumerate(ks):
                    results[k] = preds_np[i]

            # --- Inverse TTA ---
            final_accum = np.zeros_like(img_np)
            for k in range(GlobalConfig.TTA_VIEWS):
                final_accum += d4_inverse_transform(results[k], k)

            # Average over views
            final_pred = final_accum / GlobalConfig.TTA_VIEWS

            # --- Crop and Format ---
            if img_id in shapes:
                h_orig, w_orig = shapes[img_id]
                # Padding adds to bottom and right, so we crop top-left
                final_pred = final_pred[:h_orig, :w_orig]
            else:
                print(
                    f"Warning: Original shape for {img_id} not found. Using padded size."
                )

            # Flatten for CSV
            h, w = final_pred.shape
            values = final_pred.flatten()

            # Generate IDs: {img_id}_{row}_{col} (1-based)
            # Create grid of indices
            r_indices = np.repeat(np.arange(1, h + 1), w)
            c_indices = np.tile(np.arange(1, w + 1), h)

            # Construct IDs list
            ids = [f"{img_id}_{r}_{c}" for r, c in zip(r_indices, c_indices)]

            # Create DataFrame for this image
            df_out = pd.DataFrame({"id": ids, "value": values})

            # Append to CSV
            df_out.to_csv(sub_file, mode="a", header=False, index=False)

    print(f"Submission generation complete. Saved to {sub_file}")
