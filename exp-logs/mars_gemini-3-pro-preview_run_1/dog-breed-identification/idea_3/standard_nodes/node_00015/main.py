import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image

# Import provided library components
from library.config import Config
from library.utils import set_seed, calculate_log_loss
from library.dataset import load_data, get_transforms, DogDataset
from library.model import build_model
from library.engine import run_training, generate_submission


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 2. Data Loading
    # Load metadata with caching
    train_df = load_data("train", load_cached_data=True)
    val_df = load_data("val", load_cached_data=True)

    # Get transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    # Create Datasets
    train_dataset = DogDataset(train_df, transform=train_transform, mode="train")
    val_dataset = DogDataset(val_df, transform=val_transform, mode="val")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Must be False to align predictions with DataFrame for analysis
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 3. Model Building
    model = build_model(pretrained=True)
    model.to(device)

    # 4. Training
    # Execute the two-phase training pipeline
    run_training(
        model,
        train_loader,
        val_loader,
        device,
        phase1_epochs=Config.PHASE1_EPOCHS,
        phase2_epochs=Config.PHASE2_EPOCHS,
        patience=Config.PATIENCE,
    )

    # 5. Evaluation
    # Load the best model checkpoint
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    model.eval()
    val_probs = []
    val_labels = []

    # Inference on validation set with TTA (Cite solution_lesson_node_00014)
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Original
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # Flip TTA
            images_flip = torch.flip(images, dims=[3])
            outputs_flip = model(images_flip)
            probs_flip = torch.softmax(outputs_flip, dim=1)

            # Average
            avg_probs = 0.5 * (probs + probs_flip)

            val_probs.append(avg_probs.cpu().numpy())
            val_labels.append(labels.numpy())

    val_probs = np.concatenate(val_probs, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    # Calculate Metric
    metric = calculate_log_loss(val_labels, val_probs)
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    print("Performing failure analysis...")

    # Calculate per-sample loss (Cross Entropy)
    # Clip to avoid log(0)
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)

    # Get probability assigned to the true class
    # val_labels are integer indices
    true_class_probs = val_probs_clipped[np.arange(len(val_labels)), val_labels]
    losses = -np.log(true_class_probs)

    # Create analysis dataframe
    analysis_df = val_df.copy()
    analysis_df["loss"] = losses

    # Extract image metadata (width, height, aspect_ratio, area)
    widths = []
    heights = []
    aspect_ratios = []
    areas = []

    for _, row in analysis_df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 0)
                areas.append(w * h)
        except Exception:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            areas.append(0)

    analysis_df["width"] = widths
    analysis_df["height"] = heights
    analysis_df["aspect_ratio"] = aspect_ratios
    analysis_df["area"] = areas

    # Calculate and print correlations
    features = ["width", "height", "aspect_ratio", "area"]
    print("Correlation between Error Magnitude and Input Features:")
    for feature in features:
        corr = analysis_df[feature].corr(analysis_df["loss"])
        print(f"{feature}: {corr}")

    # 7. Submission
    threshold = 0.14144190501755333
    if metric < threshold:
        print("Metric check passed. Generating submission...")

        # Load test data
        test_df = load_data("test", load_cached_data=True)

        # Create test dataset and loader
        test_dataset = DogDataset(test_df, transform=val_transform, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        # Generate submission file with TTA (Cite solution_lesson_node_00014)
        generate_submission(model, test_loader, device, tta=True)
    else:
        print(
            f"Metric {metric} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
