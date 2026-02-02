import os
import sys
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import KuzushijiDataset
from library.utils import (
    get_train_transform,
    get_valid_transform,
    collate_fn,
    calculate_f1_score,
)
from library.model import get_model
from library.engine import train_eval_loop, inference, evaluate


def main():
    # Enforce reproducibility
    Config.set_seed(Config.SEED)

    # Device configuration
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Override Config for Fast Baseline / Time Constraints
    # 12 epochs should be sufficient to reach convergence without exceeding the time limit.
    Config.EPOCHS = 12

    # Load Metadata
    print("Loading metadata...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Initialize Datasets
    # We use the full provided training set (2245 images) to maximize performance.
    train_dataset = KuzushijiDataset(
        df_train,
        Config.INPUT_DIR,
        transforms=get_train_transform(),
        load_cached_data=True,
    )

    val_dataset = KuzushijiDataset(
        df_val,
        Config.INPUT_DIR,
        transforms=get_valid_transform(),
        load_cached_data=True,
    )

    # Initialize DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Initialize Model
    print("Initializing model...")
    model = get_model()
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    train_eval_loop(
        model, optimizer, train_loader, val_loader, device, epochs=Config.EPOCHS
    )

    # Load Best Model for Analysis & Inference
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using current model state.")

    # Final Validation Metric
    print("Computing final validation metrics...")
    metrics = evaluate(model, val_loader, device)
    final_f1 = metrics["f1"]
    # Required output format
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()

    f1_scores = []
    widths = []
    heights = []
    char_counts = []

    # Disable gradients for analysis
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                # Metadata
                tgt_labels = targets[i]["labels"]
                num_chars = len(tgt_labels)
                h, w = targets[i]["orig_size"].tolist()

                # Predictions (Rescale for metric calculation)
                scale = targets[i]["scale_factor"].item()

                # Prepare dicts for metric function
                # Note: We keep them as tensors on CPU as calculate_f1_score handles .cpu().numpy()
                p_dict = {
                    "boxes": output["boxes"].cpu() / scale,
                    "labels": output["labels"].cpu(),
                    "scores": output["scores"].cpu(),
                }
                t_dict = {
                    "boxes": targets[i]["boxes"].cpu() / scale,
                    "labels": targets[i]["labels"].cpu(),
                }

                # Calculate F1 for this single image
                m = calculate_f1_score([p_dict], [t_dict])

                f1_scores.append(m["f1"])
                widths.append(w)
                heights.append(h)
                char_counts.append(num_chars)

    # Calculate Correlations
    errors = 1.0 - np.array(f1_scores)
    widths = np.array(widths)
    heights = np.array(heights)
    char_counts = np.array(char_counts)

    # Handle edge case of constant input (std=0) resulting in NaN correlation
    def safe_corr(x, y):
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return np.corrcoef(x, y)[0, 1]

    corr_width = safe_corr(errors, widths)
    corr_height = safe_corr(errors, heights)
    corr_chars = safe_corr(errors, char_counts)

    print("Correlation between Error Magnitude (1 - F1) and Input Features:")
    print(f"  Image Width: {corr_width:.4f}")
    print(f"  Image Height: {corr_height:.4f}")
    print(f"  Character Count: {corr_chars:.4f}")

    # Submission Generation
    threshold = 0.8082640433856375
    if final_f1 > threshold:
        print(
            f"\nValidation F1 ({final_f1}) exceeds threshold ({threshold}). Generating submission..."
        )

        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        test_dataset = KuzushijiDataset(
            df_test,
            Config.INPUT_DIR,
            transforms=get_valid_transform(),  # No augmentation for testing
            load_cached_data=True,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Ensure output directory exists
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        inference(model, test_loader, device, output_path=submission_path)
    else:
        print(
            f"\nValidation F1 ({final_f1}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
