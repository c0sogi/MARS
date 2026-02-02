import os
import gc
import pandas as pd
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders, get_test_dataloader
from library.model_factory import build_model
from library.trainer import Trainer


def analyze_failures(val_df, val_preds, val_labels):
    """
    Performs failure analysis by correlating error magnitude with image metadata.
    """
    print("\n--- Failure Analysis ---")

    # Calculate absolute error
    errors = np.abs(val_labels - val_preds)

    # Prepare lists for metadata features
    file_sizes = []
    widths = []
    heights = []
    aspect_ratios = []

    print("Extracting metadata features for correlation analysis...")
    # Iterate through validation samples to get metadata
    # We use PIL for fast header reading of width/height
    for idx, row in val_df.iterrows():
        filepath = row["filepath"]
        full_path = os.path.join(Config.INPUT_DIR, filepath)

        if os.path.exists(full_path):
            # File Size
            file_sizes.append(os.path.getsize(full_path))

            # Dimensions
            try:
                with Image.open(full_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h if h > 0 else 0)
            except Exception:
                widths.append(0)
                heights.append(0)
                aspect_ratios.append(0)
        else:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "file_size": file_sizes,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
        }
    )

    # Compute and print correlations
    features = ["file_size", "width", "height", "aspect_ratio"]
    for feat in features:
        if analysis_df[feat].std() > 0:  # Avoid constant input
            corr, pval = pearsonr(analysis_df[feat], analysis_df["error"])
            print(
                f"Error vs {feat.ljust(15)}: Correlation = {corr: .4f} (p-value = {pval:.4f})"
            )
        else:
            print(f"Error vs {feat.ljust(15)}: N/A (Constant feature)")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()
    device = get_device()
    print(f"Running on device: {device}")

    # Storage for ensemble predictions
    # List of arrays: [model1_probs, model2_probs, ...]
    val_preds_ensemble = []
    test_preds_ensemble = []

    # Load validation metadata for alignment and analysis
    val_df = pd.read_csv(Config.VAL_METADATA)

    # 2. Iterate Models
    for arch_cfg in Config.MODEL_ARCHITECTURES:
        model_name = arch_cfg["model_name"]
        resolution = arch_cfg["resolution"]
        use_gem = arch_cfg["use_gem"]
        use_msd = arch_cfg["use_msd"]

        print(f"\n" + "=" * 50)
        print(f"Processing Model: {model_name}")
        print(f"Configuration: Resolution={resolution}, GeM={use_gem}, MSD={use_msd}")
        print("=" * 50)

        # A. Data Loading
        print(f"Loading data (Resolution: {resolution})...")
        train_loader, val_loader = get_dataloaders(
            resolution, Config.BATCH_SIZE, debug=Config.DEBUG
        )
        test_loader = get_test_dataloader(
            resolution, Config.BATCH_SIZE, debug=Config.DEBUG
        )

        # B. Model Building
        print("Building model...")
        model = build_model(
            model_name, num_classes=1, pretrained=True, use_gem=use_gem, use_msd=use_msd
        )

        # C. Optimization Setup
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # D. Training
        save_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")
        trainer = Trainer(model, optimizer, scheduler, device, save_path)

        trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

        # E. Inference (Validation)
        print("Generating validation predictions...")
        val_loss, val_probs, val_labels = trainer.evaluate(val_loader)
        val_preds_ensemble.append(val_probs.flatten())

        # F. Inference (Test)
        print("Generating test predictions (with TTA)...")
        test_probs, test_ids = trainer.predict(test_loader, use_tta=True)
        test_preds_ensemble.append(test_probs.flatten())

        # G. Cleanup
        print("Cleaning up resources...")
        del model, optimizer, scheduler, trainer, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    # 3. Ensemble Aggregation
    print("\n" + "=" * 50)
    print("Aggregating Ensemble Predictions")
    print("=" * 50)

    # Stack and average (Arithmetic Mean)
    # Shape: (Num_Models, Num_Samples) -> Mean over axis 0
    avg_val_preds = np.mean(np.stack(val_preds_ensemble), axis=0)
    avg_test_preds = np.mean(np.stack(test_preds_ensemble), axis=0)

    # 4. Validation Assessment
    # Ensure labels match predictions (val_loader is not shuffled, so order should match val_df)
    # We use the labels returned from the last trainer evaluation to be safe about order
    final_val_labels = val_labels.flatten()

    final_metric = log_loss(final_val_labels, avg_val_preds)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # Failure Analysis
    analyze_failures(val_df, avg_val_preds, final_val_labels)

    # 5. Submission
    THRESHOLD = 0.009241249605204765

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) meets threshold (< {THRESHOLD}). Generating submission..."
        )

        submission_path = Config.SUBMISSION_PATH

        # Use test_ids from the last iteration (order is consistent)
        final_test_ids = test_ids.flatten().astype(int)

        sub_df = pd.DataFrame({"id": final_test_ids, "label": avg_test_preds})

        # Sort by ID as required
        sub_df = sub_df.sort_values("id")

        # Save
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not meet threshold (< {THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
