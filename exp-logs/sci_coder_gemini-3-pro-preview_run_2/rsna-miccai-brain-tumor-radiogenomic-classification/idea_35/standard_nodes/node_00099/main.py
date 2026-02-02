import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import library modules
import library.config as C
import library.utils as U
import library.data as D
import library.model as M
import library.train as T


def main():
    # 1. Setup
    U.seed_everything(C.SEED)
    device = torch.device(C.DEVICE)
    print(f"Running on device: {device}")

    # 2. Train the model
    # We use the provided fit function which handles the training loop,
    # data loading (with caching), and saving the best model.
    print("Starting training phase...")
    T.fit(load_cached_data=True)

    # 3. Validation & Failure Analysis
    print("Starting validation and failure analysis...")

    # Load the best model
    model = M.AsymmetricEfficientNet().to(device)
    if os.path.exists(C.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(C.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Error: Model file not found.")
        return

    model.eval()

    # Load Validation Data
    # We rely on the metadata files generated previously
    if not os.path.exists(C.VAL_METADATA_PATH):
        print("Error: Validation metadata not found.")
        return

    val_df = pd.read_csv(C.VAL_METADATA_PATH)
    val_dataset = D.MGMTDataset(
        metadata_df=val_df,
        transform=D.get_transforms(phase="val"),
        cache_path=C.VAL_CACHE_PATH,
        load_cached_data=True,
        is_test=False,
    )
    val_loader = D.get_dataloader(
        val_dataset, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=C.NUM_WORKERS
    )

    # Validation Inference
    val_probs = []
    val_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            val_probs.extend(probs)
            val_labels.extend(labels.numpy().flatten())

    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)

    # Compute Metric
    try:
        # Handle case with single class in validation
        if len(np.unique(val_labels)) > 1:
            val_auc = roc_auc_score(val_labels, val_probs)
        else:
            val_auc = 0.5
    except Exception as e:
        print(f"Error computing AUC: {e}")
        val_auc = 0.5

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_auc}")

    # Failure Analysis: Correlation between Error and FLAIR Slice Count
    # Calculate absolute errors
    errors = np.abs(val_labels - val_probs)

    # Extract slice counts from metadata paths (proxy for input complexity)
    slice_counts = []
    for _, row in val_df.iterrows():
        flair_dir = os.path.join(C.INPUT_DIR, row["path_FLAIR"])
        try:
            # Count files in directory
            count = len([f for f in os.listdir(flair_dir) if f.endswith(".dcm")])
        except:
            count = 0
        slice_counts.append(count)

    slice_counts = np.array(slice_counts)

    # Compute correlation
    if len(errors) > 1 and np.std(errors) > 0 and np.std(slice_counts) > 0:
        corr_matrix = np.corrcoef(errors, slice_counts)
        correlation = corr_matrix[0, 1]
        print(
            f"Failure Analysis - Correlation between Error and FLAIR Slice Count: {correlation:.4f}"
        )
    else:
        print(
            "Failure Analysis - Correlation could not be computed (insufficient variance)."
        )

    # 4. Submission
    THRESHOLD = 0.6321818181818182

    if val_auc > THRESHOLD:
        print(
            f"Validation AUC ({val_auc:.4f}) > Threshold ({THRESHOLD:.4f}). Generating submission..."
        )

        if not os.path.exists(C.TEST_METADATA_PATH):
            print("Error: Test metadata not found.")
            return

        test_df = pd.read_csv(C.TEST_METADATA_PATH)
        test_dataset = D.MGMTDataset(
            metadata_df=test_df,
            transform=D.get_transforms(
                phase="val"
            ),  # No augmentation in dataset, done manually for TTA
            cache_path=C.TEST_CACHE_PATH,
            load_cached_data=True,
            is_test=True,
        )
        test_loader = D.get_dataloader(
            test_dataset,
            batch_size=C.BATCH_SIZE,
            shuffle=False,
            num_workers=C.NUM_WORKERS,
        )

        test_probs = []
        test_ids = []

        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                images = images.to(device)

                # TTA Strategy: Average of Original, HFlip, VFlip

                # 1. Original
                out_orig = model(images)
                prob_orig = torch.sigmoid(out_orig)

                # 2. Horizontal Flip (dim 3: W)
                images_h = torch.flip(images, dims=[3])
                out_h = model(images_h)
                prob_h = torch.sigmoid(out_h)

                # 3. Vertical Flip (dim 2: H)
                images_v = torch.flip(images, dims=[2])
                out_v = model(images_v)
                prob_v = torch.sigmoid(out_v)

                # Average probabilities
                avg_prob = (prob_orig + prob_h + prob_v) / 3.0

                test_probs.extend(avg_prob.cpu().numpy().flatten())

                # Retrieve IDs
                # Since shuffle=False, we can map indices
                start_idx = i * C.BATCH_SIZE
                end_idx = start_idx + images.size(0)
                batch_ids = test_dataset.bra_ids[start_idx:end_idx]
                test_ids.extend(batch_ids)

        # Generate Submission File
        submission = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_probs})

        submission.to_csv(C.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {C.SUBMISSION_PATH}")

    else:
        print(
            f"Validation AUC ({val_auc:.4f}) did not meet threshold ({THRESHOLD:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
