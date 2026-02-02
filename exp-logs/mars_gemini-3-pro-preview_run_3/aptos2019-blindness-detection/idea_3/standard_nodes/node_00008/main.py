import os
import cv2
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Import from provided libraries
from library.utils import seed_everything, quadratic_weighted_kappa
from library.dataset import get_dataloaders
from library.model import RetinopathyModel
from library.train import train
from library.predict import inference_fn


def evaluate_and_analyze(
    model, val_loader, device, metadata_dir="./metadata", input_dir="./input"
):
    """
    Runs inference on validation set, calculates metric, and performs failure analysis.
    """
    model.eval()
    all_preds = []
    all_labels = []

    # 1. Inference Loop
    print("Running validation inference for analysis...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # Use mixed precision for speed
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images).view(-1)

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 2. Calculate Metric
    # Clip and round for QWK
    preds_rounded = np.round(np.clip(all_preds, 0, 4)).astype(int)
    labels_int = all_labels.astype(int)
    qwk = quadratic_weighted_kappa(labels_int, preds_rounded)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {qwk}")

    # 3. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Calculate absolute error
    errors = np.abs(all_preds - all_labels)

    # Load validation metadata to get file paths
    val_df = pd.read_csv(os.path.join(metadata_dir, "val.csv"))

    # Ensure alignment: The val_loader is sequential (shuffle=False), so indices match val_df
    if len(val_df) != len(errors):
        print("Warning: Mismatch between validation set size and predictions.")
        return qwk

    # Extract meta-features
    widths = []
    heights = []
    aspect_ratios = []
    intensities = []

    print("Extracting meta-features from validation images...")
    for _, row in val_df.iterrows():
        full_path = os.path.join(input_dir, row["file_path"])
        try:
            # Read image to get original properties
            img = cv2.imread(full_path)
            if img is None:
                # Fallback for missing images
                widths.append(0)
                heights.append(0)
                aspect_ratios.append(0)
                intensities.append(0)
                continue

            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)

            # Simple mean intensity (normalized)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            intensities.append(img_rgb.mean() / 255.0)

        except Exception:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            intensities.append(0)

    # Compute Correlations
    meta_features = {
        "Width": widths,
        "Height": heights,
        "Aspect Ratio": aspect_ratios,
        "Mean Intensity": intensities,
    }

    print("Correlation between Absolute Error and Input Features (Spearman):")
    for name, values in meta_features.items():
        if len(values) == len(errors):
            corr, _ = spearmanr(errors, values)
            print(f"  {name}: {corr:.4f}")
        else:
            print(f"  {name}: Could not compute (length mismatch)")

    return qwk


def main():
    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    # Cite solution_lesson_node_00006: Increasing resolution to 768 to address error-dimension correlation.
    # Cite solution_lesson_node_00002: Increasing epochs and patience to accommodate heavier augmentation.
    BATCH_SIZE = 8
    EPOCHS = 25
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 768

    WORKING_DIR = "./working/idea_3"
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"
    METRIC_THRESHOLD = 0.9061213229890489

    print("Starting Orchestration...")

    # 2. Train Model
    # We use the library train function which handles the loop and saving best model
    print(f"Training for {EPOCHS} epochs...")
    train(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        patience=6,  # Relaxed patience for harder task
        save_dir=WORKING_DIR,
        submission_path=SUBMISSION_PATH,  # Placeholder, we generate conditionally later
    )

    # 3. Load Best Model for Analysis
    if not os.path.exists(BEST_MODEL_PATH):
        print("Error: Best model not found. Training may have failed.")
        return

    print(f"Loading best model from {BEST_MODEL_PATH}...")
    model = RetinopathyModel(
        pretrained=False
    )  # No need to download weights, we load state_dict
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)

    # 4. Get Validation Loader
    # We only need val_loader here
    _, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, load_cached_data=True
    )

    # 5. Validation & Failure Analysis
    final_metric = evaluate_and_analyze(model, val_loader, device)

    # 6. Conditional Submission
    if final_metric > METRIC_THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({METRIC_THRESHOLD:.6f}). Generating submission..."
        )
        # Reuse the loaded model and test_loader
        inference_fn(
            model_path=BEST_MODEL_PATH,
            output_path=SUBMISSION_PATH,
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            device=device,
        )
    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({METRIC_THRESHOLD:.6f}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
