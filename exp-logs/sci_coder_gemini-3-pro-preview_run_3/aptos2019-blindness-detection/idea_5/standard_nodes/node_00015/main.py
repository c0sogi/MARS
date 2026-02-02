import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2

# Import from provided libraries
from library.utils import seed_everything
from library.dataset import prepare_datasets
from library.model import get_model
from library.engine import train_loop, validate, generate_submission


def analyze_failures(model, val_loader, val_metadata_path, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and image meta-features.
    """
    print("\n--- Failure Analysis ---")

    # 1. Get Predictions and Targets
    model.eval()
    preds_list = []
    targets_list = []

    # The val_loader from prepare_datasets is sequential (shuffle=False)
    # This ensures alignment with the metadata CSV read below
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.numpy())

    preds = np.concatenate(preds_list).flatten()
    targets = np.concatenate(targets_list).flatten()

    # Calculate error magnitude (using continuous predictions vs ground truth)
    errors = np.abs(targets - preds)

    # 2. Extract Meta-features
    # Load metadata to get file paths
    if not os.path.exists(val_metadata_path):
        print(
            f"Metadata file not found at {val_metadata_path}. Skipping detailed analysis."
        )
        return

    df_val = pd.read_csv(val_metadata_path)
    input_dir = "./input"

    widths = []
    heights = []
    aspect_ratios = []
    intensities = []

    print("Extracting meta-features for validation set...")
    # Iterate through the dataframe to extract original image stats
    for idx, row in df_val.iterrows():
        file_path = os.path.join(input_dir, row["file_path"])
        try:
            # Read image to get original dimensions and intensity
            # We read the original file to see if original image properties correlate with error
            img = cv2.imread(file_path)
            if img is None:
                widths.append(0)
                heights.append(0)
                aspect_ratios.append(0)
                intensities.append(0)
                continue

            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)

            # Mean intensity (simple approximation)
            mean_intensity = img.mean() / 255.0
            intensities.append(mean_intensity)

        except Exception:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            intensities.append(0)

    # 3. Calculate Correlations
    # Using pandas to calculate Spearman correlation
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "width": widths[
                : len(errors)
            ],  # Ensure lengths match if any mismatch occurred
            "height": heights[: len(errors)],
            "aspect_ratio": aspect_ratios[: len(errors)],
            "intensity": intensities[: len(errors)],
        }
    )

    print("Correlation between Error Magnitude and Meta-features (Spearman):")
    for feat in ["width", "height", "aspect_ratio", "intensity"]:
        if feat in df_analysis.columns:
            corr = df_analysis["error"].corr(df_analysis[feat], method="spearman")
            print(f"  {feat}: Correlation = {corr:.4f}")


def main():
    # 1. Configuration
    SEED = 42
    seed_everything(SEED)

    # Hyperparameters
    # Image size 768x768 is large, so we use a small batch size
    IMAGE_SIZE = 768
    BATCH_SIZE = 4
    # Accumulate gradients to simulate a larger effective batch size (4 * 8 = 32)
    ACCUMULATION_STEPS = 8
    EPOCHS = 8
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CACHE_DIR = "./working/idea_5"
    MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"
    VAL_METADATA_PATH = "./metadata/val.csv"

    print(f"Using device: {DEVICE}")
    print(
        f"Configuration: Size={IMAGE_SIZE}, Batch={BATCH_SIZE}, Accum={ACCUMULATION_STEPS}, Epochs={EPOCHS}"
    )

    # 2. Data Preparation
    print("\n--- Data Preparation ---")
    # load_cached_data=True allows using pre-processed npy files if they exist
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        image_size=IMAGE_SIZE, load_cached_data=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,  # Inference can handle larger batches
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("\n--- Model Initialization ---")
    # ConvNeXt Base with LayerNorm is robust to small batch sizes
    model = get_model(model_name="convnext_base", pretrained=True)
    model.to(DEVICE)

    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # 5. Training Loop
    print("\n--- Starting Training ---")
    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        accumulation_steps=ACCUMULATION_STEPS,
        patience=5,
        save_path=MODEL_SAVE_PATH,
    )

    # 6. Final Evaluation & Failure Analysis
    print("\n--- Final Evaluation ---")
    # Load the best model saved during training
    model = get_model(
        model_name="convnext_base", pretrained=False, checkpoint_path=MODEL_SAVE_PATH
    )
    model.to(DEVICE)

    # Calculate final metric on validation set
    final_kappa, _ = validate(model, val_loader, DEVICE)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_kappa}")

    # Perform Failure Analysis
    analyze_failures(model, val_loader, VAL_METADATA_PATH, DEVICE)

    # 7. Submission
    THRESHOLD_METRIC = 0.9147885422881397

    if final_kappa > THRESHOLD_METRIC:
        print(
            f"\nValidation metric ({final_kappa}) meets threshold ({THRESHOLD_METRIC})."
        )
        generate_submission(model, test_loader, DEVICE, output_path=SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric ({final_kappa}) does not meet threshold ({THRESHOLD_METRIC}). Submission skipped."
        )


if __name__ == "__main__":
    main()
