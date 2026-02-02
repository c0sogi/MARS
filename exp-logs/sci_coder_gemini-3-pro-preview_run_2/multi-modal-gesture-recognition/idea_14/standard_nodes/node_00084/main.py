import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import provided library components
from library.utils import set_seed, get_device
from library.model import GLT_CRCN
from library.data_loader import get_data, GestureDataset, collate_fn, DataLoaderConfig
from library.train_eval import (
    train_model,
    generate_submission,
    evaluate_levenshtein,
    levenshtein_distance,
    post_process_sequence,
)


def perform_failure_analysis(model, val_loader, device, median_window=7):
    """
    Analyzes the correlation between model error and input characteristics.
    """
    model.eval()

    # Load Ground Truth
    val_meta_path = os.path.join(DataLoaderConfig.METADATA_DIR, "val.csv")
    df_val = pd.read_csv(val_meta_path)
    df_val["labels"] = df_val["labels"].apply(
        lambda x: (
            [int(i) for i in str(x).split()]
            if pd.notna(x) and str(x).strip() != ""
            else []
        )
    )
    gt_map = dict(zip(df_val["sample_id"], df_val["labels"]))

    errors = []
    seq_lengths = []
    num_gestures = []
    motion_energies = []

    print("\nRunning Failure Analysis...")

    with torch.no_grad():
        for feats, _, mask, ids in val_loader:
            feats, mask = feats.to(device), mask.to(device)

            # Forward pass
            outputs = model(feats, mask)
            logits = outputs[-1]
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

            # Iterate batch
            for i, sample_id in enumerate(ids):
                valid_len = int(mask[i].sum().item())

                # 1. Error Magnitude (Levenshtein Distance)
                seq_preds = preds[i][:valid_len]
                pred_seq = post_process_sequence(seq_preds, median_window=median_window)
                gt_seq = gt_map.get(sample_id, [])
                dist = levenshtein_distance(pred_seq, gt_seq)
                errors.append(dist)

                # 2. Sequence Length
                seq_lengths.append(valid_len)

                # 3. Number of Gestures
                num_gestures.append(len(gt_seq))

                # 4. Motion Energy (Mean of Velocity Magnitude)
                # Features: [Pos(36), Vel(36), Audio(13)]
                # Velocity channels are 36 to 71
                # feats is (B, C, T)
                vel_feats = feats[i, 36:72, :valid_len]  # (36, T)
                # L2 norm over channels, then mean over time
                energy = torch.norm(vel_feats, p=2, dim=0).mean().item()
                motion_energies.append(energy)

    # Compute Correlations
    if len(errors) > 1:
        corr_len, _ = pearsonr(errors, seq_lengths)
        corr_num, _ = pearsonr(errors, num_gestures)
        corr_mot, _ = pearsonr(errors, motion_energies)

        print(f"Correlation (Error vs Seq Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")
        print(f"Correlation (Error vs Motion Energy): {corr_mot:.4f}")
    else:
        print("Insufficient data for correlation analysis.")


def main():
    # 1. Setup
    set_seed(42)
    device = get_device()
    working_dir = "./working/idea_14"
    os.makedirs(working_dir, exist_ok=True)

    print(f"Running on device: {device}")

    # 2. Train Model (Fast Baseline)
    # We use 15 epochs which is sufficient for convergence on this dataset with A100
    # while keeping runtime low.
    best_model_path = train_model(
        epochs=50,
        batch_size=16,
        lr=1e-3,
        weight_decay=1e-4,
        smoothing_weight=0.15,
        boundary_weight=1.0,
        patience=10,
        median_window=7,
        augment=True,
        working_dir=working_dir,
        load_cached_data=True,
    )

    # 3. Final Evaluation
    print("Loading best model for evaluation...")
    model = GLT_CRCN().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Load Validation Data
    val_data = get_data("val", load_cached_data=True)
    val_dataset = GestureDataset(val_data, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Compute Metric
    final_metric = evaluate_levenshtein(model, val_loader, device, median_window=7)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    perform_failure_analysis(model, val_loader, device, median_window=7)

    # 5. Submission
    threshold = 0.08548168249660787
    if final_metric < threshold:
        print(
            f"Validation metric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )
        generate_submission(
            model_path=best_model_path,
            median_window=7,
            submission_dir="./submission",
            load_cached_data=True,
        )
    else:
        print(
            f"Validation metric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
