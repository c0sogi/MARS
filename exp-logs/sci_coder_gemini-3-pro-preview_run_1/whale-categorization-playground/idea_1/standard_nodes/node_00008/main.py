import os
import cv2
import torch
import numpy as np
import pandas as pd
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.model import WhaleClassifier, validate, generate_predictions
from library.train import train_model


def run():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    DATA_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_FILE = "./submission/submission.csv"

    # Hyperparameters for a fast baseline execution
    # 8 epochs is sufficient for ResNet18 to learn features on this dataset size
    # while keeping runtime well under the limit.
    # Cite solution_lesson_node_00003: Extended training for scheduler and augmentation
    EPOCHS = 15
    BATCH_SIZE = 32
    LR = 1e-4
    PATIENCE = 4
    # Cite solution_lesson_node_00005: Increased resolution to preserve fine-grained details
    IMAGE_SIZE = 320

    # Ensure reproducibility
    set_seed(42)

    print("=== Starting Orchestration ===")

    # ---------------------------------------------------------
    # 2. Train Model & Generate Submission
    # ---------------------------------------------------------
    # This function orchestrates data loading, training, early stopping,
    # and generates the submission file at the end.
    train_model(
        data_dir=DATA_DIR,
        metadata_dir=METADATA_DIR,
        working_dir=WORKING_DIR,
        submission_file=SUBMISSION_FILE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        patience=PATIENCE,
        image_size=IMAGE_SIZE,
        load_cached_data=True,  # Use cached label encoding if available
    )

    # ---------------------------------------------------------
    # 3. Validation Assessment
    # ---------------------------------------------------------
    print("\n=== Starting Validation Analysis ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Re-initialize loaders to access the validation set specifically.
    # We use the same cache_dir to ensure the LabelEncoder mapping is identical.
    _, val_loader, test_loader, label_encoder = get_dataloaders(
        data_dir=DATA_DIR,
        metadata_dir=METADATA_DIR,
        batch_size=BATCH_SIZE,
        load_cached_data=True,
        cache_dir=WORKING_DIR,
        image_size=IMAGE_SIZE,
    )

    # Initialize model architecture
    num_classes = label_encoder.num_classes()
    model = WhaleClassifier(num_classes=num_classes)

    # Load the best checkpoint saved during training
    checkpoint_path = os.path.join(WORKING_DIR, "model_best.pth.tar")
    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("Error: Checkpoint not found. Cannot perform validation analysis.")
        return

    model = model.to(device)
    model.eval()

    # Calculate Final Metric on the Hold-out Validation Set
    criterion = torch.nn.CrossEntropyLoss()
    # validate() returns (avg_loss, map5)
    _, final_map5 = validate(val_loader, model, criterion, device)

    # Print the required metric string
    print(f"Final Validation Metric: {final_map5}")

    # Conditional Submission Generation
    if final_map5 > 0.6279379157:
        print(f"Metric {final_map5} > threshold. Generating submission...")
        generate_predictions(
            model=model,
            test_loader=test_loader,
            label_encoder=label_encoder,
            device=device,
            output_file=SUBMISSION_FILE,
        )
    else:
        print(f"Metric {final_map5} <= threshold. Skipping submission generation.")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Failure Analysis ===")

    # 4.1 Calculate Per-Sample Error
    # Error is defined as (1.0 - Sample_MAP_Score).
    # If the correct label is rank 1, score is 1.0, error is 0.0.
    # If the correct label is not in top 5, score is 0.0, error is 1.0.

    sample_errors = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            # Get top 5 predictions
            _, preds = outputs.topk(5, 1, True, True)

            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()

            for i in range(len(targets)):
                t = targets[i]
                p = preds[i]

                if t in p:
                    rank = np.where(p == t)[0][0]
                    score = 1.0 / (rank + 1.0)
                else:
                    score = 0.0

                sample_errors.append(1.0 - score)

    # 4.2 Extract Input Features
    # The val_loader iterates sequentially (shuffle=False).
    # We load the validation metadata to get file paths and extract image features.
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))

    if len(df_val) != len(sample_errors):
        print(
            f"Warning: Validation set size mismatch. DF: {len(df_val)}, Preds: {len(sample_errors)}"
        )

    widths = []
    heights = []
    intensities = []

    # Read images to get dimensions and intensity
    for idx, row in df_val.iterrows():
        full_path = os.path.join(DATA_DIR, row["file_path"])

        # Read image
        img = cv2.imread(full_path, cv2.IMREAD_COLOR)

        if img is None:
            # Fallback
            widths.append(0)
            heights.append(0)
            intensities.append(0)
        else:
            h, w = img.shape[:2]
            # Normalize intensity to [0, 1]
            mean_intensity = img.mean() / 255.0

            widths.append(w)
            heights.append(h)
            intensities.append(mean_intensity)

    # 4.3 Compute Correlations
    errors_arr = np.array(sample_errors)
    widths_arr = np.array(widths)
    heights_arr = np.array(heights)
    intensities_arr = np.array(intensities)

    def safe_corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return np.corrcoef(a, b)[0, 1]

    corr_w = safe_corr(errors_arr, widths_arr)
    corr_h = safe_corr(errors_arr, heights_arr)
    corr_i = safe_corr(errors_arr, intensities_arr)

    print(f"Correlation (Error vs Width): {corr_w:.6f}")
    print(f"Correlation (Error vs Height): {corr_h:.6f}")
    print(f"Correlation (Error vs Intensity): {corr_i:.6f}")

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    run()
