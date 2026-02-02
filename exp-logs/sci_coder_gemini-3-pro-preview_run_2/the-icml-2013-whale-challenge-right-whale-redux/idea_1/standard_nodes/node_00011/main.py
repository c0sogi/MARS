import os
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.model import WhaleResNet
from library.dataset import get_dataloaders, WhaleDataset
from library.trainer import train_model, generate_submission


def perform_failure_analysis(model, val_loader, val_dataset):
    """
    Analyzes validation errors and correlates them with input features.
    """
    print("\n--- Performing Failure Analysis ---")
    device = torch.device(Config.DEVICE)
    model.eval()

    all_probs = []
    all_labels = []

    # 1. Get Model Predictions on Validation Set
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 2. Calculate Error Magnitude
    # Error is the absolute difference between probability and ground truth (0 or 1)
    errors = np.abs(all_probs - all_labels)

    # 3. Extract Audio Features for Correlation Analysis
    # We iterate through the dataset to extract signal properties
    durations = []
    rms_values = []
    peaks = []

    print(f"Extracting audio features for {len(val_dataset)} validation samples...")

    df = val_dataset.df
    root_dir = val_dataset.root_dir

    for _, row in df.iterrows():
        full_path = os.path.join(root_dir, row["file_path"])
        try:
            # Read audio metadata and signal
            info = sf.info(full_path)
            audio, sr = sf.read(full_path)

            # Convert to mono if necessary
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Calculate features
            dur = info.duration
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))

            durations.append(dur)
            rms_values.append(rms)
            peaks.append(peak)

        except Exception:
            # Fallback for unreadable files (should be rare/non-existent based on metadata check)
            durations.append(0.0)
            rms_values.append(0.0)
            peaks.append(0.0)

    # 4. Compute Correlations
    # Create a DataFrame for easy correlation computation
    analysis_df = pd.DataFrame(
        {"error": errors, "duration": durations, "rms": rms_values, "peak": peaks}
    )

    print("\nCorrelation between Error Magnitude and Input Features:")
    features = ["duration", "rms", "peak"]
    for feat in features:
        # Compute Pearson correlation
        if analysis_df[feat].std() > 0 and analysis_df["error"].std() > 0:
            corr = np.corrcoef(analysis_df["error"], analysis_df[feat])[0, 1]
        else:
            corr = 0.0
        print(f"{feat.capitalize()}: {corr:.4f}")


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Set seeds for reproducibility
    Config.set_seed()

    # Configure for a fast baseline run
    # Using Config values directly (optimized in config.py)

    # Create directories
    Config.setup()

    print(
        f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # ==========================================
    # 2. Model Training
    # ==========================================
    print("\n--- Starting Training ---")
    # train_model trains the model, saves the best checkpoint, and returns the model with best weights
    model = train_model(num_epochs=Config.NUM_EPOCHS)

    # ==========================================
    # 3. Validation Evaluation
    # ==========================================
    print("\n--- Validating Model ---")
    # Retrieve dataloaders (re-using the function from library)
    _, val_loader, _ = get_dataloaders()

    device = torch.device(Config.DEVICE)
    model.to(device)
    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(labels.numpy().flatten())

    # Compute Final Metric
    val_auc = roc_auc_score(all_targets, all_probs)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    # Instantiate dataset wrapper to access file paths for feature extraction
    val_dataset = WhaleDataset(Config.VAL_CSV, Config.INPUT_ROOT, is_test=False)
    perform_failure_analysis(model, val_loader, val_dataset)

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    print("\n--- Generating Submission ---")
    generate_submission(model)


if __name__ == "__main__":
    main()
