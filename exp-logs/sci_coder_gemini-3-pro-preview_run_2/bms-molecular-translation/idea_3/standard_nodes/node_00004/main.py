import sys
import os
import torch
import pandas as pd
import numpy as np
import cv2
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append("./")

from library.config import Config
from library.utils import seed_everything, calc_levenshtein
from library.dataset import get_dataloaders
from library.model import CNNTransformer
from library.trainer import Trainer
from library.inference import run_inference
from library.tokenizer import InChITokenizer


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for a fast baseline run within time limits
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 128  # Maximize for A100 GPU
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10000  # Sufficient for baseline training

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    seed_everything(Config.SEED)

    print("=" * 40)
    print("ORCHESTRATION SCRIPT STARTED")
    print(f"Device: {device}")
    print(f"Training Samples: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print("=" * 40)

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\nLoading Data...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=Config.DEBUG,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("Initializing Model...")
    model = CNNTransformer().to(device)

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("\nStarting Training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=Config.LEARNING_RATE,
        patience=Config.PATIENCE,
    )

    trainer.fit(epochs=Config.EPOCHS)

    # ---------------------------------------------------------
    # 5. Validation Assessment & Failure Analysis
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("VALIDATION & FAILURE ANALYSIS")
    print("=" * 40)

    model.eval()
    tokenizer = InChITokenizer()

    val_dists = []
    val_widths = []
    val_heights = []
    val_target_lens = []

    # Access the dataframe used by the validation loader to get file paths
    df_val = val_loader.dataset.df

    print(f"Evaluating on {len(df_val)} validation samples...")

    with torch.no_grad():
        batch_idx = 0
        # Iterate through loader to handle preprocessing/batching
        for batch in val_loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            # Predict (Greedy Decoding)
            pred_indices = model.predict(images, max_len=Config.MAX_LEN, device=device)

            # Process batch
            for k in range(len(pred_indices)):
                # Decode texts
                pred_str = tokenizer.decode(pred_indices[k])
                true_str = tokenizer.decode(labels[k])

                # Compute metric for this sample
                dist = calc_levenshtein([pred_str], [true_str])
                val_dists.append(dist)
                val_target_lens.append(len(true_str))

                # Retrieve image metadata
                # Map batch index back to dataframe index
                # Note: val_loader is not shuffled, so order is preserved
                global_idx = batch_idx * Config.BATCH_SIZE + k

                if global_idx < len(df_val):
                    file_path = df_val.iloc[global_idx]["file_path"]
                    full_path = os.path.join(Config.INPUT_DIR, file_path)

                    # We need dimensions. Reading image headers is fast.
                    # If reading is too slow, we default to 0.
                    try:
                        # OpenCV imread is reasonably fast
                        img = cv2.imread(full_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            val_widths.append(w)
                            val_heights.append(h)
                        else:
                            val_widths.append(0)
                            val_heights.append(0)
                    except:
                        val_widths.append(0)
                        val_heights.append(0)
                else:
                    val_widths.append(0)
                    val_heights.append(0)

            batch_idx += 1

    # Compute Final Metric
    final_metric = np.mean(val_dists)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n[Failure Analysis]")
    if len(val_dists) > 1:
        # Filter valid image stats
        valid_indices = [i for i, w in enumerate(val_widths) if w > 0]

        if len(valid_indices) > 1:
            clean_dists = [val_dists[i] for i in valid_indices]
            clean_widths = [val_widths[i] for i in valid_indices]
            clean_heights = [val_heights[i] for i in valid_indices]
            clean_lens = [val_target_lens[i] for i in valid_indices]

            # Correlations
            corr_w, _ = pearsonr(clean_dists, clean_widths)
            corr_h, _ = pearsonr(clean_dists, clean_heights)
            corr_len, _ = pearsonr(clean_dists, clean_lens)

            print("Correlations with Levenshtein Distance:")
            print(f"  Image Width:   {corr_w:.4f}")
            print(f"  Image Height:  {corr_h:.4f}")
            print(f"  Target Length: {corr_len:.4f}")

            if abs(corr_len) > 0.3:
                print("  -> Significant correlation with target length observed.")
        else:
            print("  -> Insufficient valid image metadata for correlation analysis.")
    else:
        print("  -> Not enough validation samples for analysis.")

    # ---------------------------------------------------------
    # 6. Submission Generation
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("GENERATING SUBMISSION")
    print("=" * 40)

    # Run inference on the FULL test set (debug=False)
    # We use the same batch size as training/val
    run_inference(
        checkpoint_path=Config.MODEL_SAVE_PATH,
        output_path=Config.SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        device=str(device),
        debug=False,
    )

    print("\nPipeline Completed Successfully.")


if __name__ == "__main__":
    main()
