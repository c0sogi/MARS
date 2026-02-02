import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from the provided library
import library.config as config
from library.dataset import get_dataloaders
from library.engine import Trainer
from library.utils import set_seed, calculate_accuracy


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model failures on the validation set by correlating
    error magnitude with input signal features.
    """
    print("\n" + "=" * 30)
    print("FAILURE ANALYSIS")
    print("=" * 30)

    model.eval()

    # 1. Collect Predictions and Errors
    errors = []
    filepaths = []

    # Access the underlying dataframe to get filepaths and labels
    val_df = val_loader.dataset.df

    # We need to iterate the loader to get model probabilities
    # Note: val_loader is not shuffled, so order matches val_df
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu())
            all_targets.append(labels.cpu())

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    # Calculate Error Magnitude: 1.0 - Probability of the True Class
    # Gather the probability assigned to the correct label
    true_class_probs = all_probs.gather(1, all_targets.view(-1, 1)).squeeze()
    error_magnitudes = 1.0 - true_class_probs.numpy()

    # 2. Extract Features for Validation Files
    print("Extracting audio features for validation set...")
    feature_data = {"duration": [], "rms": [], "zcr": [], "spectral_centroid": []}

    # Iterate through the dataframe to load raw audio
    # We limit to a subset if it's too large, but 12k is manageable in ~2 mins on modern CPU
    input_root = config.INPUT_ROOT

    for idx, row in val_df.iterrows():
        full_path = os.path.join(input_root, row["filepath"])
        try:
            wav, sr = torchaudio.load(full_path)

            # Duration
            duration = wav.shape[1] / sr

            # Convert to Mono for stats
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0)
            else:
                wav = wav.squeeze(0)

            # RMS
            rms = torch.sqrt(torch.mean(wav**2)).item()

            # Zero Crossing Rate
            zcr = ((wav[:-1] * wav[1:]) < 0).sum().item() / max(1, len(wav))

            # Spectral Centroid (Approximate using FFT)
            # Simple weighted average of frequencies
            fft = torch.fft.rfft(wav)
            magnitudes = torch.abs(fft)
            freqs = torch.fft.rfftfreq(len(wav), 1 / sr)
            if magnitudes.sum() > 0:
                spec_cent = (magnitudes * freqs).sum() / magnitudes.sum()
                spec_cent = spec_cent.item()
            else:
                spec_cent = 0.0

            feature_data["duration"].append(duration)
            feature_data["rms"].append(rms)
            feature_data["zcr"].append(zcr)
            feature_data["spectral_centroid"].append(spec_cent)

        except Exception as e:
            # Fallback for read errors
            feature_data["duration"].append(1.0)
            feature_data["rms"].append(0.0)
            feature_data["zcr"].append(0.0)
            feature_data["spectral_centroid"].append(0.0)

    # 3. Calculate Correlations
    print("\nCorrelation between Error Magnitude and Input Features:")
    print(f"{'Feature':<20} | {'Pearson r':<10} | {'P-value':<10}")
    print("-" * 46)

    for feature_name, values in feature_data.items():
        # Ensure lengths match (they should)
        if len(values) != len(error_magnitudes):
            print(f"Skipping {feature_name}: Length mismatch.")
            continue

        corr, p_val = pearsonr(error_magnitudes, values)
        print(f"{feature_name:<20} | {corr:+.4f}     | {p_val:.4f}")


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Baseline
    # -------------------------------------------------------------------------
    # We modify the global dictionary directly so imported modules see the changes
    print("Configuring Fast Baseline parameters...")
    config.TRAINING_PARAMS["epochs"] = 15
    config.TRAINING_PARAMS["swa_start_epoch"] = 11
    config.TRAINING_PARAMS["target_sample_count"] = 1000  # Reduce upsampling for speed

    set_seed(config.TRAINING_PARAMS["seed"])

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config.TRAINING_PARAMS["batch_size"],
        num_workers=config.TRAINING_PARAMS["num_workers"],
        load_cached_data=True,
    )

    # -------------------------------------------------------------------------
    # 3. Training
    # -------------------------------------------------------------------------
    trainer = Trainer(train_loader, val_loader, test_loader)
    trainer.fit()

    # -------------------------------------------------------------------------
    # 4. Validation & Metric
    # -------------------------------------------------------------------------
    print("\nEvaluating Final SWA Model on Validation Set...")
    # Get the trained SWA model
    swa_model = trainer.swa_handler.get_averaged_model()

    # Run evaluation
    val_loss, val_acc = trainer.evaluate(val_loader, model_to_eval=swa_model)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_acc}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    perform_failure_analysis(swa_model, val_loader, trainer.device)

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.9872909698996656

    if val_acc > THRESHOLD:
        print(
            f"\nValidation metric ({val_acc}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict_and_submit()
    else:
        print(
            f"\nValidation metric ({val_acc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
