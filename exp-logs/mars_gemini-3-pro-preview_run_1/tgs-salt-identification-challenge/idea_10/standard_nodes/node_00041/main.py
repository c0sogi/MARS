import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided libraries
from library.utils import set_seed, do_kaggle_metric
from library.model import DeepResUNet
from library.dataset import preprocess_and_cache, SaltDataset
from library.train import Trainer
from library.inference import predict_and_submit


def run_validation_and_analysis(model, device, cache_dir):
    """
    Runs inference on the validation set to compute the final metric
    and perform failure analysis.
    """
    print("Running validation and failure analysis...")

    # Load validation data from cache (generated during training)
    # mode='val_src' was used in get_loaders inside Trainer, but here we can load
    # the split logic. However, to be precise and consistent with the Trainer's split,
    # we should rely on the fact that Trainer used a 90/10 split on the combined data.
    # Replicating the exact split here might be tricky if we don't use the exact same seed/logic.
    #
    # To ensure we evaluate on the exact same validation set the Trainer used,
    # we can re-use the get_loaders function from library.dataset or manually replicate the split.
    # The library.dataset.get_loaders function does the split.

    from library.dataset import get_loaders

    _, val_loader = get_loaders(
        train_metadata_path="./metadata/train.csv",
        val_metadata_path="./metadata/val.csv",
        cache_dir=cache_dir,
        batch_size=32,
        num_workers=4,
        load_cached_data=True,
        debug=False,
    )

    model.eval()
    all_preds = []
    all_targets = []
    all_z = []
    all_coverage = []

    # We need to link predictions back to metadata for failure analysis.
    # The loader returns (images, masks). It doesn't return IDs or metadata directly.
    # However, the split is deterministic (seed 42).
    # We will reconstruct the metadata for the validation set.

    # Re-load all metadata to reconstruct the split indices
    ids_1, _, _, classes_1 = preprocess_and_cache(
        "./metadata/train.csv", cache_dir, mode="train_src"
    )
    ids_2, _, _, classes_2 = preprocess_and_cache(
        "./metadata/val.csv", cache_dir, mode="val_src"
    )

    # Load raw dfs to get 'z' and 'coverage'
    df_train = pd.read_csv("./metadata/train.csv")
    df_val = pd.read_csv("./metadata/val.csv")

    # Concatenate in the same order as get_loaders
    all_ids = np.concatenate([ids_1, ids_2])
    all_classes = np.concatenate([classes_1, classes_2])

    # Merge dfs to get z and coverage map
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    # Map ID to z and coverage
    id_to_meta = df_all.set_index("id")[["z", "coverage"]].to_dict("index")

    # Perform split indices
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(all_ids))
    _, val_idx = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=all_classes
    )

    val_ids_ordered = all_ids[val_idx]

    # Collect predictions
    ptr = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            logits = outputs["logits"]
            probs = torch.sigmoid(logits)

            # Unpad: 128 -> 101. Indices [13:114]
            final_probs_cropped = probs[:, :, 13:114, 13:114]
            masks_cropped = masks[:, :, 13:114, 13:114]

            batch_preds = final_probs_cropped.cpu().numpy()
            batch_targets = masks_cropped.cpu().numpy()

            # Squeeze channel
            if batch_preds.ndim == 4:
                batch_preds = batch_preds.squeeze(1)
            if batch_targets.ndim == 4:
                batch_targets = batch_targets.squeeze(1)

            all_preds.append(batch_preds)
            all_targets.append(batch_targets)

            # Collect metadata for this batch
            batch_size = images.size(0)
            batch_ids = val_ids_ordered[ptr : ptr + batch_size]
            ptr += batch_size

            for bid in batch_ids:
                meta = id_to_meta.get(bid, {"z": 0, "coverage": 0})
                all_z.append(meta["z"])
                all_coverage.append(meta["coverage"])

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_z = np.array(all_z)
    all_coverage = np.array(all_coverage)

    # --- Calculate Metric ---
    # do_kaggle_metric calculates mean over the batch
    final_metric = do_kaggle_metric(all_preds, all_targets, threshold=0.5)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Calculate per-image AP to correlate with metadata
    # We replicate the metric logic per image
    per_image_scores = []
    thresholds = np.arange(0.5, 1.0, 0.05)

    bin_preds = (all_preds > 0.5).astype(np.uint8)
    bin_targets = (all_targets > 0.5).astype(np.uint8)

    for i in range(len(all_preds)):
        p = bin_preds[i]
        t = bin_targets[i]

        intersection = np.sum((p == 1) & (t == 1))
        union = np.sum((p == 1) | (t == 1))

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        matches = iou > thresholds
        score = np.mean(matches)
        per_image_scores.append(score)

    per_image_scores = np.array(per_image_scores)
    error_magnitude = 1.0 - per_image_scores

    # Correlations
    # Handle NaNs if any (though data analysis showed none)
    corr_z, _ = pearsonr(error_magnitude, all_z)
    corr_cov, _ = pearsonr(error_magnitude, all_coverage)

    print("Failure Analysis:")
    print(f"  Correlation (Error vs Depth 'z'): {corr_z:.4f}")
    print(f"  Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    return final_metric


def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories
    input_dir = "./input"
    working_dir = "./working"
    cache_dir = os.path.join(working_dir, "idea_10")
    checkpoint_dir = os.path.join(working_dir, "checkpoints")

    # 2. Training
    # We use 150 epochs to complete three full cosine cycles (T_0=50).
    # This enables Snapshot Ensembling (Cite solution_lesson_node_00035).
    print("Initializing training...")
    trainer = Trainer(
        epochs=150,
        batch_size=32,
        num_workers=4,
        lr=1e-3,
        checkpoint_dir=checkpoint_dir,
        cache_dir=cache_dir,
        debug=False,  # Use full dataset
    )

    trainer.fit()

    # 3. Validation & Failure Analysis
    print("Loading best model for validation...")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found.")
        return

    model = DeepResUNet(in_channels=2, classes=1).to(device)
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)

    metric = run_validation_and_analysis(model, device, cache_dir)

    # 4. Submission
    if metric > 0.833:
        print("Metric threshold met. Generating submission...")
        # predict_and_submit handles loading checkpoints (prioritizing cycles if present, else best_model)
        # Since we ran 50 epochs, cycle_2 (100) and cycle_3 (150) won't exist.
        # It will correctly fallback to best_model.pth.
        predict_and_submit(
            checkpoint_dir=checkpoint_dir,
            output_dir="./submission",
            cache_dir=cache_dir,
            batch_size=32,
            device=device,
        )
    else:
        print(f"Metric {metric} did not meet threshold 0.833. Skipping submission.")


if __name__ == "__main__":
    main()
