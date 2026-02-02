import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.dataset import get_dataloaders
from library.trainer import train_model_for_seed, inference
from library.utils import set_seed, calculate_auc
from library.layers import ContextGatedResNet18


def extract_analysis_features(filepath):
    """
    Extracts simple audio features for failure analysis.
    """
    try:
        waveform, sr = torchaudio.load(filepath)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze().numpy()

        # RMS Energy
        rms = np.sqrt(np.mean(waveform**2))

        # Zero Crossing Rate
        if len(waveform) > 1:
            zcr = ((waveform[:-1] * waveform[1:]) < 0).mean()
        else:
            zcr = 0.0

        # Spectral Flatness
        fft = np.fft.rfft(waveform)
        magnitude = np.abs(fft)
        mag_sq = magnitude**2
        if np.mean(mag_sq) > 0:
            gmean = np.exp(np.mean(np.log(mag_sq + 1e-10)))
            amean = np.mean(mag_sq)
            flatness = gmean / (amean + 1e-10)
        else:
            flatness = 0.0

        return rms, zcr, flatness
    except Exception:
        return 0.0, 0.0, 0.0


def main():
    # 1. Configuration
    # Cite solution_lesson_node_00066: Ensembling reduces variance.
    # We use the extended seed list from Config.

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    print(
        f"Training with Batch Size: {Config.BATCH_SIZE}, Epochs per seed: {Config.EPOCHS}"
    )

    # 2. Load Data
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # Load validation metadata for analysis
    df_val = pd.read_csv(Config.VAL_CSV)
    val_targets = df_val["label"].values

    # 3. Ensemble Training
    seeds = Config.SEEDS
    val_probs_ensemble = np.zeros(len(df_val))
    test_probs_ensemble = np.zeros(len(test_loader.dataset))

    # To store test clips order
    test_clips_ordered = None

    print(f"Starting Ensemble Training with {len(seeds)} seeds...")

    for seed in seeds:
        # Train
        train_model_for_seed(seed, train_loader, val_loader, device)

        # Load Best Model for Inference
        model = ContextGatedResNet18(config=Config).to(device)
        model_path = os.path.join(Config.OUTPUT_DIR, f"model_seed_{seed}.pth")

        if not os.path.exists(model_path):
            print(f"Warning: Model for seed {seed} not found. Skipping.")
            continue

        model.load_state_dict(torch.load(model_path, map_location=device))

        # Inference on Validation
        print(f"Running inference on Validation Set for Seed {seed}...")
        _, val_probs = inference(model, val_loader, device)
        val_probs_ensemble += np.array(val_probs)

        # Inference on Test
        print(f"Running inference on Test Set for Seed {seed}...")
        test_clips, test_probs = inference(model, test_loader, device)
        test_probs_ensemble += np.array(test_probs)

        if test_clips_ordered is None:
            test_clips_ordered = test_clips

    # Average Predictions
    val_probs_ensemble /= len(seeds)
    test_probs_ensemble /= len(seeds)

    # 4. Final Validation Metric
    final_auc = calculate_auc(val_targets, val_probs_ensemble)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(val_targets - val_probs_ensemble)

    print("Extracting features for failure analysis on validation set...")
    analysis_data = []

    for idx, row in df_val.iterrows():
        filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])
        rms, zcr, flatness = extract_analysis_features(filepath)
        analysis_data.append(
            {"rms": rms, "zcr": zcr, "flatness": flatness, "error": errors[idx]}
        )

    df_analysis = pd.DataFrame(analysis_data)

    if not df_analysis.empty:
        print("Correlation between Error Magnitude and Input Features:")
        for feat in ["rms", "zcr", "flatness"]:
            corr, _ = pearsonr(df_analysis["error"], df_analysis[feat])
            print(f"{feat}: {corr:.4f}")

    # 6. Submission Logic
    threshold = 0.9956103812188066
    if final_auc > threshold:
        print(
            f"\nValidation AUC ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        df_sub = pd.DataFrame(
            {"clip": test_clips_ordered, "probability": test_probs_ensemble}
        )

        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nValidation AUC ({final_auc}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
