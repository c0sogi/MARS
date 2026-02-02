import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.utils import seed_everything, calculate_roc_auc, load_metadata, get_device
from library.trainer import Trainer
from library.dataset import BraTSDataset


def analyze_failures(val_df, y_true, y_pred):
    """
    Performs failure analysis by correlating prediction error with metadata features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate error
    errors = np.abs(np.array(y_true) - np.array(y_pred))

    # Extract structural features (slice counts)
    # We use the paths in val_df to count files
    input_dir = "./input"
    flair_counts = []
    t2w_counts = []

    for _, row in val_df.iterrows():
        # Count FLAIR
        p_flair = os.path.join(input_dir, row["path_FLAIR"])
        if os.path.exists(p_flair):
            flair_counts.append(len(os.listdir(p_flair)))
        else:
            flair_counts.append(0)

        # Count T2w
        p_t2w = os.path.join(input_dir, row["path_T2w"])
        if os.path.exists(p_t2w):
            t2w_counts.append(len(os.listdir(p_t2w)))
        else:
            t2w_counts.append(0)

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {"error": errors, "flair_slices": flair_counts, "t2w_slices": t2w_counts}
    )

    # Compute correlations
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations.drop("error"))  # Drop self-correlation

    return correlations


def main():
    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Running on device: {device}")

    # Configuration for Fast Baseline
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4

    # 2. Training
    print("\n--- Starting Training ---")
    trainer = Trainer(learning_rate=LEARNING_RATE, device=str(device))

    # Fit the model
    # We use the full training set but limited epochs for speed
    trainer.fit(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=4,
        patience=5,
        load_cached_data=True,
    )

    # 3. Validation & Metric
    print("\n--- Performing Final Validation ---")

    # Load best model weights
    if os.path.exists(trainer.best_model_path):
        trainer.model.load_state_dict(
            torch.load(trainer.best_model_path, map_location=device)
        )
        print("Loaded best model weights.")
    else:
        print("Warning: Best model not found. Using current weights.")

    trainer.model.eval()

    # Setup Validation Data
    val_dataset = BraTSDataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_preds = []
    all_labels = []

    # Inference Loop
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = trainer.model(images)
            probs = torch.sigmoid(outputs)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    # Calculate Metric
    final_auc = calculate_roc_auc(all_labels, all_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    # Load metadata dataframe to map back to features
    val_df = load_metadata("val")
    analyze_failures(val_df, all_labels, all_preds)

    # 5. Submission
    THRESHOLD = 0.6254545454545455

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict_with_tta(
            batch_size=BATCH_SIZE, num_workers=4, load_cached_data=True
        )
    else:
        print(
            f"\nValidation metric ({final_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
