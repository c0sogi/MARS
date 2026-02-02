import os
import cv2
import torch
import numpy as np
import pandas as pd
import nltk
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.dataset import ChemicalDataset
from library.model import CRNN
from library.engine import fit, predict_and_submit, collate_fn
from library.tokenizer import Tokenizer


def main():
    print("Initializing Baseline Pipeline...")

    # ---------------------------------------------------------
    # 1. Configuration for Fast Baseline
    # ---------------------------------------------------------
    # Override Config attributes for a fast but representative baseline run.
    # We train on a subset (100k samples) for 5 epochs.
    Config.EPOCHS = 5
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100000
    Config.BATCH_SIZE = 64
    Config.NUM_WORKERS = 4

    print(f"Configuration:")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Debug Mode: {Config.DEBUG}")
    print(f"  Training Samples: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Device: {Config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("STARTING TRAINING")
    print("=" * 40)

    # fit() handles loading data, training loop, and saving the best model.
    best_model_path = fit(epochs=Config.EPOCHS)
    print(f"Training finished. Best model saved to: {best_model_path}")

    # ---------------------------------------------------------
    # 3. Full Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("STARTING VALIDATION & ANALYSIS")
    print("=" * 40)

    # IMPORTANT: Disable DEBUG mode to load the FULL validation set for final metrics
    Config.DEBUG = False

    # Load Model
    device = Config.DEVICE
    model = CRNN().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    tokenizer = Tokenizer()

    # Load Full Validation Dataset
    print("Loading full validation dataset...")
    val_dataset = ChemicalDataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,  # Inference can use larger batches
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Validating on {len(val_dataset)} samples...")

    all_distances = []
    target_lengths = []

    # We collect image features for a subset to perform failure analysis efficiently
    subset_size = 1000
    subset_indices = set(
        np.random.choice(
            len(val_dataset), size=min(subset_size, len(val_dataset)), replace=False
        )
    )
    subset_data = {"width": [], "height": [], "aspect_ratio": [], "levenshtein": []}

    global_idx = 0

    with torch.no_grad():
        for batch_idx, (images, targets, lengths) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            # Forward
            log_probs = model(images)

            # Greedy Decode
            preds = torch.argmax(log_probs, dim=2)
            decoded_preds = tokenizer.decode_batch(preds)

            # Decode Targets (handle padding)
            targets_cpu = targets.cpu()
            decoded_targets = []
            for i in range(batch_size):
                l = lengths[i].item()
                seq = targets_cpu[i][:l].tolist()
                t_str = "".join([tokenizer.idx2char.get(idx, "") for idx in seq])
                decoded_targets.append(t_str)

            # Compute Metrics
            for i, (p, t) in enumerate(zip(decoded_preds, decoded_targets)):
                dist = nltk.edit_distance(p, t)
                all_distances.append(dist)
                target_lengths.append(len(t))

                # If sample is in analysis subset, read image dimensions
                current_idx = global_idx + i
                if current_idx in subset_indices:
                    row = val_dataset.df.iloc[current_idx]
                    full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
                    try:
                        img = cv2.imread(full_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            subset_data["width"].append(w)
                            subset_data["height"].append(h)
                            subset_data["aspect_ratio"].append(w / h)
                            subset_data["levenshtein"].append(dist)
                    except Exception:
                        pass

            global_idx += batch_size

    # Final Metric
    final_metric = np.mean(all_distances)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # Failure Analysis Correlations
    print("\n[Failure Analysis]")

    # 1. Target Length vs Error (on full val set)
    corr_len, _ = pearsonr(target_lengths, all_distances)
    print(f"Correlation (Target Length vs Error): {corr_len:.4f}")

    # 2. Image Features vs Error (on subset)
    if len(subset_data["width"]) > 10:
        corr_w, _ = pearsonr(subset_data["width"], subset_data["levenshtein"])
        corr_h, _ = pearsonr(subset_data["height"], subset_data["levenshtein"])
        corr_ar, _ = pearsonr(subset_data["aspect_ratio"], subset_data["levenshtein"])

        print(f"Correlation (Image Width vs Error): {corr_w:.4f}")
        print(f"Correlation (Image Height vs Error): {corr_h:.4f}")
        print(f"Correlation (Aspect Ratio vs Error): {corr_ar:.4f}")
    else:
        print("Insufficient data for image feature correlation analysis.")

    # ---------------------------------------------------------
    # 4. Submission
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("GENERATING SUBMISSION")
    print("=" * 40)

    # predict_and_submit handles loading test data and saving predictions
    # Config.DEBUG is False, so it will process the full test set.
    predict_and_submit(best_model_path)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
