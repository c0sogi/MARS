import os
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from sklearn.metrics import roc_auc_score
import warnings

# Import provided library modules
from library.config import Config
from library.trainer import ModelTrainer
from library.dataset import get_dataloaders
from library.model import ShallowCNN

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_audio_stats(df):
    """
    Extracts basic signal statistics (Mean Amplitude, Std Dev)
    from the raw audio files referenced in the dataframe.
    Used for failure analysis.
    """
    mean_amps = []
    std_amps = []

    print(f"Extracting audio features for {len(df)} files...")

    for idx, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # Read audio file
            data, sr = sf.read(file_path)

            # Convert to mono if necessary
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            # Calculate stats
            mean_amps.append(np.mean(np.abs(data)))
            std_amps.append(np.std(data))

        except Exception as e:
            # Fallback for unreadable files
            mean_amps.append(0.0)
            std_amps.append(0.0)

    return mean_amps, std_amps


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("--- Setting up Configuration ---")

    # Override Config for a fast baseline execution
    # We use the full dataset (DEBUG=False) but limit epochs to 5 for speed.
    # The A100 GPU can handle the full dataset (approx 18k samples) quickly.
    Config.EPOCHS = 5
    Config.DEBUG = False

    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    print("\n--- Starting Training ---")
    # Initialize trainer with modified epochs
    trainer = ModelTrainer(epochs=Config.EPOCHS, debug=Config.DEBUG)

    # Train the model
    best_model_path = trainer.train()

    # -------------------------------------------------------------------------
    # 3. Validation & Metrics
    # -------------------------------------------------------------------------
    print("\n--- Performing Validation ---")

    # Retrieve the validation loader from the trainer
    val_loader = trainer.dataloaders["val"]

    # Load the best model for inference
    model = ShallowCNN().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    # Run inference on validation set
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Collect results
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(labels.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute and print the required metric
    val_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {val_auc}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Performing Failure Analysis ---")

    # Load validation metadata to link predictions with file properties
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment
    if len(val_df) == len(all_preds):
        # Attach predictions and labels
        val_df["probability"] = all_preds
        val_df["label"] = all_targets

        # Calculate Error Magnitude
        val_df["error"] = np.abs(val_df["label"] - val_df["probability"])

        # Extract input features (Audio Signal Stats)
        # We correlate error with signal amplitude to see if volume affects performance
        mean_amps, std_amps = extract_audio_stats(val_df)
        val_df["mean_amp"] = mean_amps
        val_df["std_amp"] = std_amps

        # Calculate Correlations
        corr_mean = val_df["error"].corr(val_df["mean_amp"])
        corr_std = val_df["error"].corr(val_df["std_amp"])

        print("Correlation between Model Error and Input Features:")
        print(f"  Mean Amplitude: {corr_mean:.6f}")
        print(f"  Std Amplitude:  {corr_std:.6f}")
    else:
        print("Error: Validation metadata length mismatch. Skipping detailed analysis.")

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    print("\n--- Generating Submission ---")
    trainer.generate_submission(best_model_path)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
