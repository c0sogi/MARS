import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.dataset import RetinopathyDataset
from library.model import RetinopathyModel
from library.utils import seed_everything, ModelEMA
from library.engine import train_one_epoch, evaluate, predict_tta


def get_validation_meta_features(df):
    """
    Extracts original image dimensions and file sizes for failure analysis.
    """
    widths = []
    heights = []
    file_sizes = []

    for _, row in df.iterrows():
        file_path = os.path.join(Config.input_root, row["file_path"])
        try:
            # File size
            if os.path.exists(file_path):
                file_sizes.append(os.path.getsize(file_path))

                # Dimensions - read using cv2
                img = cv2.imread(file_path)
                if img is not None:
                    h, w, _ = img.shape
                    widths.append(w)
                    heights.append(h)
                else:
                    widths.append(0)
                    heights.append(0)
            else:
                file_sizes.append(0)
                widths.append(0)
                heights.append(0)
        except Exception:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)

    return np.array(widths), np.array(heights), np.array(file_sizes)


def main():
    # 1. Initialization
    seed_everything(Config.seed)

    # Override Config to resolve OOM and bypass Module Caching
    print(f"Overriding Batch Size: Old={Config.train_batch_size}, New=8")
    Config.train_batch_size = 8
    Config.val_batch_size = 8

    os.makedirs("./submission", exist_ok=True)

    print(f"Starting execution with Device: {Config.device}")
    print(f"Batch Size set to: {Config.train_batch_size}")

    # 2. Data Loading
    print("Loading Metadata...")
    train_df = pd.read_csv(Config.train_metadata_path)
    val_df = pd.read_csv(Config.val_metadata_path)
    test_df = pd.read_csv(Config.test_metadata_path)

    print("Initializing Datasets (with caching)...")
    train_dataset = RetinopathyDataset(train_df, phase="train", load_cached_data=True)
    val_dataset = RetinopathyDataset(val_df, phase="val", load_cached_data=True)
    test_dataset = RetinopathyDataset(test_df, phase="test", load_cached_data=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
    )

    # 3. Model Setup
    print("Initializing Model...")
    model = RetinopathyModel().to(Config.device)

    ema = None
    if Config.use_ema:
        ema = ModelEMA(model, decay=Config.ema_decay, device=Config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    # 4. Training Loop
    best_qwk = -1.0
    best_model_path = os.path.join(Config.working_dir, "best_model.pth")

    print("Starting Training Loop...")
    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=Config.device,
            ema=ema,
            scheduler=scheduler,
        )

        scheduler.step()

        # Validation
        # Use EMA model for validation if available
        val_model = ema.ema if ema else model
        val_qwk, val_loss = evaluate(val_model, val_loader, Config.device)

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val QWK: {val_qwk:.5f}"
        )

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(val_model.state_dict(), best_model_path)

    # 5. Final Assessment
    print("\nLoading Best Model for Final Assessment...")
    final_model = RetinopathyModel().to(Config.device)
    final_model.load_state_dict(torch.load(best_model_path, map_location=Config.device))
    final_model.eval()

    # Calculate final metric on validation set
    final_val_qwk, _ = evaluate(final_model, val_loader, Config.device)
    print(f"Final Validation Metric: {final_val_qwk}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Get predictions using TTA for robust analysis
    val_preds = predict_tta(final_model, val_loader, Config.device)
    val_targets = val_df["diagnosis"].values

    # Error Magnitude
    errors = np.abs(val_targets - val_preds)

    # Meta Features
    print("Extracting validation meta-features...")
    widths, heights, sizes = get_validation_meta_features(val_df)
    aspect_ratios = np.divide(
        widths, heights, out=np.zeros_like(widths, dtype=float), where=heights != 0
    )

    # Correlations
    print("Correlation between Error Magnitude and Input Features:")
    feature_map = {
        "Width": widths,
        "Height": heights,
        "Aspect Ratio": aspect_ratios,
        "File Size": sizes,
        "Ground Truth Class": val_targets,
    }

    for name, data in feature_map.items():
        if len(data) == len(errors):
            if np.std(data) > 0 and np.std(errors) > 0:
                corr, _ = pearsonr(errors, data)
                print(f"  {name}: {corr:.4f}")
            else:
                print(f"  {name}: NaN (No variance)")

    # 7. Submission
    threshold = 0.9234606183435836
    if final_val_qwk > threshold:
        print(
            f"\nFinal Metric ({final_val_qwk}) > Threshold ({threshold}). Generating Submission..."
        )

        test_preds = predict_tta(final_model, test_loader, Config.device)

        submission_df = pd.DataFrame(
            {"id_code": test_df["id_code"], "diagnosis": test_preds}
        )

        sub_path = "./submission/submission.csv"
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"\nFinal Metric ({final_val_qwk}) <= Threshold ({threshold}). Submission Skipped."
        )


if __name__ == "__main__":
    main()
