import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
from PIL import Image

# Import from library
from library.config import Config
from library.utils import set_seed, get_device, load_checkpoint
from library.train import run_fold_training
from library.dataset import get_dataloaders
from library.model import get_model
from library.inference import generate_ensemble_submission


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Training Loop (5 Folds)
    # Using 20 epochs to ensure execution finishes well within limits while maintaining high performance.
    training_epochs = 20

    print(
        f"Starting 5-Fold Training with {training_epochs} fine-tuning epochs per fold..."
    )

    for fold_idx in range(Config.N_FOLDS):
        _ = run_fold_training(
            fold_idx=fold_idx,
            warmup_epochs=Config.WARMUP_EPOCHS,
            finetune_epochs=training_epochs,
            lr_finetune=Config.LR,
            weight_decay=Config.WEIGHT_DECAY,
            patience=5,
            debug=False,
        )

    # 3. OOF Validation & Metric Calculation
    print("\nStarting Out-Of-Fold (OOF) Validation...")

    oof_preds = []
    oof_targets = []

    # Metadata storage for failure analysis
    meta_stats = {"file_size": [], "width": [], "height": [], "aspect_ratio": []}

    # Per-sample loss tracking
    all_losses = []
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    for fold_idx in range(Config.N_FOLDS):
        print(f"Validating Fold {fold_idx}...")

        # Load Data
        # load_cached_data=True ensures we use the exact same splits
        _, val_loader, classes = get_dataloaders(fold_idx, load_cached_data=True)

        # Load Model
        model = get_model(Config.MODEL_NAME, Config.NUM_CLASSES, pretrained=False)
        ckpt_path = os.path.join(Config.OUTPUT_DIR, f"model_fold_{fold_idx}.pth")

        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint for fold {fold_idx} not found. Skipping.")
            continue

        load_checkpoint(model, None, ckpt_path)
        model.to(device)
        model.eval()

        # Inference containers for this fold
        fold_preds = []
        fold_targets = []

        # Extract metadata for failure analysis
        # We iterate the dataframe associated with the validation loader
        val_df = val_loader.dataset.df

        for idx, row in val_df.iterrows():
            full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            try:
                # File size
                size = os.path.getsize(full_path)
                # Image dims (lazy load with PIL is faster than cv2.imread)
                with Image.open(full_path) as img:
                    w, h = img.size
            except Exception:
                size = 0
                w, h = 224, 224  # Fallback

            meta_stats["file_size"].append(size)
            meta_stats["width"].append(w)
            meta_stats["height"].append(h)
            meta_stats["aspect_ratio"].append(w / h if h > 0 else 0)

        # Run Inference
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                # Calculate per-sample loss
                batch_losses = criterion(outputs, labels)

                fold_preds.append(probs.cpu().numpy())
                fold_targets.append(labels.cpu().numpy())
                all_losses.extend(batch_losses.cpu().numpy())

        oof_preds.append(np.concatenate(fold_preds))
        oof_targets.append(np.concatenate(fold_targets))

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Concatenate all folds
    if not oof_preds:
        print("Error: No predictions generated.")
        return

    y_pred = np.concatenate(oof_preds)
    y_true = np.concatenate(oof_targets)
    losses = np.array(all_losses)

    # Calculate Final Metric
    final_metric = log_loss(y_true, y_pred, labels=list(range(Config.NUM_CLASSES)))

    print(f"Final Validation Metric: {final_metric:.16f}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    df_meta = pd.DataFrame(meta_stats)
    df_meta["loss"] = losses

    print("Correlation between Error (Log Loss) and Input Features:")
    for col in ["file_size", "width", "height", "aspect_ratio"]:
        if df_meta[col].std() > 0:
            corr, _ = pearsonr(df_meta["loss"], df_meta[col])
            print(f"  {col}: {corr:.4f}")
        else:
            print(f"  {col}: 0.0000")

    # 5. Submission
    threshold = 0.14004325100369866
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric:.6f}) < threshold ({threshold:.6f}). Generating submission..."
        )
        generate_ensemble_submission()
    else:
        print(
            f"\nMetric ({final_metric:.6f}) >= threshold ({threshold:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
