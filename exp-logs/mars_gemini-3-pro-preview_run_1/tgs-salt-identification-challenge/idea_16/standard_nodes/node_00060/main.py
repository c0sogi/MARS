import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.train import train_model
from library.inference import predict
from library.dataset import get_loaders
from library.model import DeepResUNet
from library.utils import set_seed, compute_salt_metric


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------------
    # Using default configuration from library.config (150 epochs, 3 cycles of 50).
    # This aligns with Lesson 00023 (Long Constant Cycles) and Lesson 00045 (Training Duration).
    # Cite {solution_lesson_node_00023}, {solution_lesson_node_00045}

    # Setup directories and seeds
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    print("--- Starting Training Pipeline ---")
    # This function handles data loading, model initialization, and the training loop
    train_model()

    # -------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load Validation Data
    # We use load_cached_data=True to leverage the .npy files created during training
    _, val_loader, _ = get_loaders(load_cached_data=True)

    # Load Models for Ensemble (Cycle 2 and Cycle 3)
    # The strategy uses an ensemble of the best snapshots from the Lovasz optimization phase
    models = []
    for cycle in [2, 3]:
        path = os.path.join(Config.CHECKPOINT_DIR, f"best_cycle_{cycle}.pth")
        if os.path.exists(path):
            print(f"Loading checkpoint: {path}")
            m = DeepResUNet().to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)
        else:
            print(f"Warning: Checkpoint for Cycle {cycle} not found.")

    # Fallback to global best if cycle models are missing
    if not models:
        print("No cycle checkpoints found. Attempting to load global best model.")
        path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(path):
            m = DeepResUNet().to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)

    if not models:
        print("Error: No trained models found. Exiting.")
        return

    # Load Metadata for Analysis (Depth 'z')
    df_val = pd.read_csv(Config.VAL_CSV)
    id_to_z = dict(zip(df_val["id"], df_val["z"]))

    # Performance-Gated Ensembling (Cite {solution_lesson_node_00050})
    # We evaluate models individually to decide whether to ensemble or select the best single model.

    model_scores = {i: [] for i in range(len(models))}
    val_maps_ensemble = []
    val_ious = []
    val_depths = []
    val_coverages = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, masks, ids in val_loader:
            images = images.to(device)
            masks = masks.to(device)  # Shape: (B, 1, H, W)

            batch_probs = []

            # Get predictions for each model
            for m_idx, model in enumerate(models):
                logits = model(images)
                probs = torch.sigmoid(logits)

                # Test-Time Augmentation (Horizontal Flip)
                if Config.TTA_FLIP:
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flipped = model(images_flipped)
                    probs_flipped = torch.sigmoid(logits_flipped)
                    probs_flipped_back = torch.flip(probs_flipped, dims=[3])
                    probs = 0.5 * (probs + probs_flipped_back)

                batch_probs.append(probs)

            # Calculate Ensemble Probabilities
            avg_probs = torch.stack(batch_probs).mean(dim=0)

            # Move to CPU
            masks_np = masks.cpu().numpy()
            avg_probs_np = avg_probs.cpu().numpy()

            # Individual model probs to CPU
            batch_probs_np = [p.cpu().numpy() for p in batch_probs]

            # Process batch
            for i in range(len(ids)):
                img_id = ids[i]
                t = masks_np[i, 0]
                t_bin = (t > 0.5).astype(np.uint8)

                # 1. Individual Scores
                for m_idx in range(len(models)):
                    p = batch_probs_np[m_idx][i, 0]
                    p_bin = (p > 0.5).astype(np.uint8)
                    score = compute_salt_metric(p_bin, t_bin)
                    model_scores[m_idx].append(score)

                # 2. Ensemble Score
                p_ens = avg_probs_np[i, 0]
                p_ens_bin = (p_ens > 0.5).astype(np.uint8)
                score_ens = compute_salt_metric(p_ens_bin, t_bin)
                val_maps_ensemble.append(score_ens)

                # 3. IoU & Metadata (using ensemble)
                intersection = np.logical_and(p_ens_bin, t_bin).sum()
                union = np.logical_or(p_ens_bin, t_bin).sum()
                iou = (
                    1.0
                    if (union == 0 and t_bin.sum() == 0)
                    else (intersection / union if union > 0 else 0.0)
                )
                val_ious.append(iou)

                z = id_to_z.get(img_id, 0)
                val_depths.append(z)
                cov = t_bin.sum() / t_bin.size
                val_coverages.append(cov)

    # Analyze Scores
    mean_scores = [np.mean(model_scores[i]) for i in range(len(models))]
    ensemble_score = np.mean(val_maps_ensemble)

    print(f"Model Scores: {mean_scores}")
    print(f"Ensemble Score: {ensemble_score}")

    # Gating Logic (Cite {solution_lesson_node_00050})
    # If the gap between models is > 0.005, discard the weak one.
    final_metric = ensemble_score

    if len(models) == 2:
        score_diff = abs(mean_scores[0] - mean_scores[1])
        best_idx = np.argmax(mean_scores)
        best_single_score = mean_scores[best_idx]

        print(f"Score Difference: {score_diff:.6f}")

        if score_diff > 0.005:
            print("Performance gap > 0.005. Discarding weaker model.")
            final_metric = best_single_score

            # Identify which cycle corresponds to the weaker model
            # models list order depends on loading order: Cycle 2 then Cycle 3
            # We assume models[0] is Cycle 2, models[1] is Cycle 3 based on loading code
            weak_idx = 1 - best_idx
            cycles = [2, 3]
            weak_cycle = cycles[weak_idx]

            # Remove the checkpoint
            chk_path = os.path.join(
                Config.CHECKPOINT_DIR, f"best_cycle_{weak_cycle}.pth"
            )
            if os.path.exists(chk_path):
                print(f"Deleting checkpoint: {chk_path}")
                os.remove(chk_path)
            else:
                print(f"Checkpoint not found for deletion: {chk_path}")
        else:
            print("Models are comparable. Keeping ensemble.")

    print(f"Final Validation Metric: {final_metric:.10f}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis Results
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis Report ---")
    errors = 1.0 - np.array(val_ious)
    depths = np.array(val_depths)
    coverages = np.array(val_coverages)

    # Correlation: Error vs Depth
    if np.std(errors) > 0 and np.std(depths) > 0:
        corr_depth, _ = pearsonr(errors, depths)
        print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    else:
        print("Correlation (Error vs Depth): N/A (Constant values)")

    # Correlation: Error vs Salt Coverage
    if np.std(errors) > 0 and np.std(coverages) > 0:
        corr_cov, _ = pearsonr(errors, coverages)
        print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")
    else:
        print("Correlation (Error vs Salt Coverage): N/A (Constant values)")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.833
    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric:.4f}) exceeds threshold ({THRESHOLD})."
        )
        print("Generating submission for test set...")
        # predict() uses the Checkpoints saved in Config.CHECKPOINT_DIR
        # It automatically ensembles Cycle 2 and Cycle 3 if present.
        predict()
    else:
        print(
            f"\nValidation metric ({final_metric:.4f}) does NOT exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
