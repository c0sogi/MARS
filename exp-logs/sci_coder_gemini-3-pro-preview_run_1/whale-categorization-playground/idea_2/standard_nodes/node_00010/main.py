import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, map5, get_id_encoder, apk
from library.data import get_loaders, WhaleDataset, get_transforms
from library.model import WhaleEfficientNet
from library.train import run_training


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)

    # 2. Training
    # We use 10 epochs to ensure a good balance between speed and convergence.
    # The A100 GPU can handle this workload efficiently.
    print("Starting training phase...")
    run_training(epochs=10, debug=False)

    # 3. Load Best Model for Validation
    print("Loading best model for validation...")
    device = Config.DEVICE
    model = WhaleEfficientNet(pretrained=False)

    # Load checkpoint
    if not os.path.exists(Config.CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {Config.CHECKPOINT_PATH}")

    checkpoint = torch.load(
        Config.CHECKPOINT_PATH, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # 4. Validation Inference
    # We need to get the validation loader.
    # Note: get_loaders returns (train_loader, val_loader)
    _, val_loader = get_loaders(debug=False, load_cached_data=True)

    all_targets = []
    all_preds = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # labels are tensors of indices

            logits = model(images)
            # Get top 5 predictions
            _, top5_indices = logits.topk(5, dim=1, largest=True, sorted=True)

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(top5_indices.cpu().numpy().tolist())

    # 5. Compute and Print Validation Metric
    val_map5 = map5(all_targets, all_preds)
    print(f"Final Validation Metric: {val_map5}")

    # 6. Failure Analysis
    print("Performing failure analysis...")

    # Calculate per-sample error (Error = 1.0 - AP)
    # apk returns the Average Precision for a single sample
    sample_aps = [apk(t, p, k=5) for t, p in zip(all_targets, all_preds)]
    errors = [1.0 - ap for ap in sample_aps]

    # Load validation metadata to access original image files
    df_val = pd.read_csv(Config.VAL_CSV)

    # Extract image features
    widths = []
    heights = []
    aspect_ratios = []
    intensities = []

    for _, row in df_val.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        # Read image to get dimensions and intensity
        # Using cv2.IMREAD_COLOR
        img = cv2.imread(full_path)

        if img is None:
            # Fallback if image read fails
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            intensities.append(0)
        else:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            # Simple mean intensity
            intensities.append(img.mean())

    # Calculate correlations
    features = {
        "Width": widths,
        "Height": heights,
        "AspectRatio": aspect_ratios,
        "Intensity": intensities,
    }

    errors_arr = np.array(errors)

    for name, vals in features.items():
        vals_arr = np.array(vals)
        if np.std(vals_arr) > 0 and np.std(errors_arr) > 0:
            corr = np.corrcoef(errors_arr, vals_arr)[0, 1]
        else:
            corr = 0.0
        print(f"Correlation between Error and {name}: {corr:.8f}")

    # 7. Submission Generation
    TARGET_THRESHOLD = 0.6306356245

    if val_map5 > TARGET_THRESHOLD:
        print(
            f"Validation metric {val_map5} exceeds threshold {TARGET_THRESHOLD}. Generating submission..."
        )

        # Load Test Metadata
        df_test = pd.read_csv(Config.TEST_CSV)

        # Get ID Encoder
        id_encoder = get_id_encoder(load_cached_data=True)

        # Create Test Dataset and Loader
        # We use 'val' transforms which are deterministic (Resize + Normalize)
        test_dataset = WhaleDataset(
            df_test,
            transforms=get_transforms("val"),
            id_encoder=id_encoder,
            is_test=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        submission_data = []

        with torch.no_grad():
            for images, image_filenames in test_loader:
                images = images.to(device)

                # TTA: Horizontal Flip
                logits1 = model(images)
                logits2 = model(torch.flip(images, dims=[3]))
                logits = (logits1 + logits2) / 2.0

                _, top5_indices = logits.topk(5, dim=1, largest=True, sorted=True)

                top5_indices = top5_indices.cpu().numpy()

                for filename, indices in zip(image_filenames, top5_indices):
                    # Convert indices back to whale IDs
                    pred_labels = id_encoder.inverse_transform(indices)
                    pred_string = " ".join(pred_labels)

                    submission_data.append({"Image": filename, "Id": pred_string})

        # Create DataFrame and save
        df_submission = pd.DataFrame(submission_data)
        df_submission.to_csv(Config.FINAL_SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.FINAL_SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric {val_map5} does not meet threshold {TARGET_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
