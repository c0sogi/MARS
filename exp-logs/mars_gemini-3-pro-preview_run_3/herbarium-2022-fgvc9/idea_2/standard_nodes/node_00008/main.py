import os
import cv2
import torch
import pandas as pd
import numpy as np
from library.utils import set_seed, calculate_macro_f1
from library.dataset import create_dataloaders
from library.trainer import Trainer


def main():
    # 1. Setup
    set_seed(42)

    # Configuration
    # Using 3 epochs to ensure the pipeline completes within the 2-hour target
    # while allowing sufficient convergence for EfficientNetV2-S on the full dataset.
    EPOCHS = 3
    BATCH_SIZE = 64

    # 2. Data Loading
    print("Initializing DataLoaders...")
    # debug=False ensures we train on the full dataset to maximize F1 score
    train_loader, val_loader, test_loader = create_dataloaders(
        train_batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE,
        num_workers=4,
        debug=False,
        img_size=260,
    )

    # 3. Training
    print("Initializing Trainer...")
    config = {"lr": 1e-3, "weight_decay": 1e-2, "patience": 5, "use_mixup": True}
    trainer = Trainer(config)

    print(f"Starting Training for {EPOCHS} epochs...")
    trainer.fit(train_loader, val_loader, epochs=EPOCHS)

    # 4. Validation with Best Model
    print("Loading best model for final validation...")
    best_model_path = os.path.join(trainer.working_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=trainer.device)
        trainer.model.load_state_dict(state_dict)
    else:
        print("Warning: Best model not found. Using current model state.")

    trainer.model.eval()

    all_preds = []
    all_labels = []

    print("Running validation inference...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(trainer.device)

            # Inference (using mixed precision for speed)
            with torch.cuda.amp.autocast():
                outputs = trainer.model(images)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Calculate Metric
    val_f1 = calculate_macro_f1(all_labels, all_preds)
    print(f"Final Validation Metric: {val_f1}")

    # 5. Failure Analysis
    print("Performing Failure Analysis...")
    val_df = pd.read_csv("./metadata/val.csv")

    # Ensure alignment between loader predictions and metadata
    if len(val_df) != len(all_preds):
        print(
            f"Error: Metadata length ({len(val_df)}) != Predictions length ({len(all_preds)})"
        )
    else:
        val_df["pred"] = all_preds
        val_df["label"] = all_labels
        val_df["is_error"] = (val_df["pred"] != val_df["label"]).astype(int)

        # Sample for image feature extraction to keep analysis fast
        sample_size = 1000
        analysis_df = val_df.sample(
            n=min(len(val_df), sample_size), random_state=42
        ).copy()

        widths = []
        heights = []
        file_sizes = []

        input_dir = "./input"

        for _, row in analysis_df.iterrows():
            path = os.path.join(input_dir, row["file_path"])
            try:
                # Get file size
                fsize = os.path.getsize(path)
                file_sizes.append(fsize)

                # Get dimensions
                img = cv2.imread(path)
                if img is not None:
                    h, w, _ = img.shape
                    widths.append(w)
                    heights.append(h)
                else:
                    widths.append(0)
                    heights.append(0)
            except Exception:
                widths.append(0)
                heights.append(0)
                file_sizes.append(0)

        analysis_df["width"] = widths
        analysis_df["height"] = heights
        analysis_df["file_size"] = file_sizes

        # Calculate correlations
        corr_width = analysis_df["is_error"].corr(analysis_df["width"])
        corr_height = analysis_df["is_error"].corr(analysis_df["height"])
        corr_size = analysis_df["is_error"].corr(analysis_df["file_size"])

        print("Correlation between Error and Input Features:")
        print(f"  Error vs Width: {corr_width:.4f}")
        print(f"  Error vs Height: {corr_height:.4f}")
        print(f"  Error vs File Size: {corr_size:.4f}")

    # 6. Submission
    THRESHOLD = 0.5930838412243743
    if val_f1 > THRESHOLD:
        print(f"Validation F1 ({val_f1}) > {THRESHOLD}. Generating submission...")
        trainer.predict(test_loader)
    else:
        print(f"Validation F1 ({val_f1}) <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
