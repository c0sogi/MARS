import os
import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config, seed_everything
from library.train import run_training_phase
from library.inference import generate_submission, extract_embeddings_with_tta
from library.dataset import (
    load_cached_images,
    WhaleDataset,
    get_transforms,
    get_label_mapping,
)
from library.model import WhaleModel
from library.utils import mapk, apk


def main():
    # 1. Setup & Configuration Override for Fast Baseline
    seed_everything(Config.SEED)

    # Override STAGES to ensure fast execution (Limit epochs and resolution)
    # We use a single stage of 256x256 for 2 epochs.
    Config.STAGES = [{"resolution": 256, "epochs": 2}]

    print("Configuration overridden for fast baseline: 256x256, 2 Epochs.")

    # 2. Run Training Phase
    # This will train both models defined in Config.MODEL_CONFIGS
    run_training_phase()

    # 3. Ensemble Validation
    print("\nStarting Ensemble Validation...")
    device = Config.DEVICE
    resolution = Config.STAGES[-1]["resolution"]

    # Load Validation Metadata and Images
    df_val = pd.read_csv(Config.VAL_CSV)
    val_images = load_cached_images(
        df_val, resolution, "val_images", load_cached_data=True
    )

    # Prepare Validation Dataset/Loader
    val_dataset = WhaleDataset(
        images=val_images,
        targets=None,  # Targets handled separately for metric calc
        transform=get_transforms("val", resolution),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Prepare Targets for Metric Calculation
    id2idx, idx2id = get_label_mapping()
    val_targets_raw = df_val["Id"].values
    # Map 'new_whale' to -1, others to their index
    val_targets_mapped = [id2idx.get(x, -1) for x in val_targets_raw]

    # Compute Ensemble Similarity Matrix
    ensemble_sim_matrix = None
    models_processed = 0

    for model_cfg in Config.MODEL_CONFIGS:
        model_name = model_cfg["name"]
        backbone = model_cfg["backbone"]
        emb_size = model_cfg["embedding_size"]

        checkpoint_path = os.path.join(
            Config.WORKING_DIR, f"{model_name}_{resolution}_best.pth"
        )

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found for {model_name}, skipping.")
            continue

        print(f"Processing validation for {model_name}...")

        # Load Model
        model = WhaleModel(backbone, pretrained=False, embedding_size=emb_size).to(
            device
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Extract Embeddings (with TTA)
        embeddings = extract_embeddings_with_tta(val_loader, model, device)

        # Extract Class Centers from Loss
        loss_state = checkpoint["loss_state_dict"]
        centers = loss_state["weight"].cpu()
        centers = F.normalize(centers, p=2, dim=1)

        # Compute Similarity
        sim_matrix = torch.matmul(embeddings, centers.T)

        if ensemble_sim_matrix is None:
            ensemble_sim_matrix = sim_matrix
        else:
            ensemble_sim_matrix += sim_matrix

        models_processed += 1

        # Cleanup
        del model, checkpoint, embeddings, centers, sim_matrix
        torch.cuda.empty_cache()

    if ensemble_sim_matrix is None:
        print("Error: No models processed for validation.")
        return

    # Average the similarities
    ensemble_sim_matrix /= models_processed

    # Generate Predictions and Calculate Metrics
    predicted_labels = []
    actual_labels = []
    ap_scores = []  # Store AP per sample for failure analysis

    threshold = Config.CONFIDENCE_THRESHOLD

    for i in range(len(val_targets_raw)):
        actual = val_targets_raw[i]
        actual_labels.append([actual])

        sims = ensemble_sim_matrix[i]
        scores, indices = torch.topk(sims, k=5)
        scores = scores.numpy()
        indices = indices.numpy()

        preds = []
        new_whale_added = False

        for score, idx in zip(scores, indices):
            # Open-set logic
            if not new_whale_added and score < threshold:
                preds.append("new_whale")
                new_whale_added = True

            if len(preds) >= 5:
                break

            preds.append(idx2id[idx])

        if len(preds) < 5 and not new_whale_added:
            preds.append("new_whale")

        preds = preds[:5]
        predicted_labels.append(preds)

        # Calculate AP for this specific sample
        ap = apk([actual], preds, k=5)
        ap_scores.append(ap)

    # Compute MAP@5
    final_metric = mapk(actual_labels, predicted_labels, k=5)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Gather Metadata features
    widths = []
    heights = []
    file_sizes = []

    for _, row in df_val.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path))
            img = cv2.imread(full_path)
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
            else:
                widths.append(0)
                heights.append(0)
        else:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)

    # Error Magnitude = 1.0 - AP
    errors = 1.0 - np.array(ap_scores)

    # Calculate Correlations
    # We use Pearson correlation
    if len(errors) > 1 and np.std(errors) > 0:
        corr_width, _ = pearsonr(errors, widths)
        corr_height, _ = pearsonr(errors, heights)
        corr_size, _ = pearsonr(errors, file_sizes)

        print(f"Correlation Error vs Width: {corr_width:.4f}")
        print(f"Correlation Error vs Height: {corr_height:.4f}")
        print(f"Correlation Error vs FileSize: {corr_size:.4f}")
    else:
        print("Insufficient variance in errors to compute correlation.")

    # 5. Conditional Submission
    submission_threshold = 0.846985
    if final_metric > submission_threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({submission_threshold}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({submission_threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
