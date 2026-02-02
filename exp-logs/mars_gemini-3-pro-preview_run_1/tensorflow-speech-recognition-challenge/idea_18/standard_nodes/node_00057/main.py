import os
import shutil
import torch
import pandas as pd
import numpy as np
import soundfile as sf
import torchaudio
import warnings

# Import library modules
from library.config import Config
from library.trainer import Trainer
from library.dataset import get_dataloaders
from library.utils import set_seed, compute_accuracy

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Initializing Orchestration Script...")

    # 1. Setup Directories
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # Ensure reproducibility
    set_seed(Config.SEED)

    # 2. Initialize Trainer and Train
    # The Trainer handles the training loop, SWA, and initial submission generation in WORK_DIR
    # We use the default Config (50 epochs) to ensure we hit the high accuracy threshold.
    # On an A100, this will complete well within the time limit.
    trainer = Trainer()
    trainer.train(load_cached_data=True)

    # 3. Independent Validation for Metric & Failure Analysis
    print("\n" + "=" * 30)
    print("POST-TRAINING VALIDATION & ANALYSIS")
    print("=" * 30)

    device = trainer.device
    model = trainer.swa_model
    model.eval()

    # Load Validation Data
    # We use the same loader function but only need val_loader
    _, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    all_probs = []
    all_targets = []
    all_filepaths = val_loader.dataset.df["filepath"].tolist()

    # Inference Loop
    print("Running inference on validation set...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            # Convert logits to probabilities
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    # 4. Calculate Final Metric
    # compute_accuracy works on logits or probabilities (argmax is invariant)
    final_acc = compute_accuracy(all_probs, all_targets)
    print(f"Final Validation Metric: {final_acc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate Error Magnitude (1 - Probability of True Class)
    # Gather prob for the correct class
    target_indices = all_targets.long()
    row_indices = torch.arange(len(all_targets)).long()

    true_class_probs = all_probs[row_indices, target_indices]
    error_magnitudes = 1.0 - true_class_probs.numpy()

    # Extract Audio Features for Correlation Analysis
    feature_records = []

    # Pre-initialize transform (expects 16k sample rate)
    spec_centroid_fn = torchaudio.transforms.SpectralCentroid(sample_rate=Config.SR)

    print(f"Extracting features for {len(all_filepaths)} validation files...")

    for i, rel_path in enumerate(all_filepaths):
        full_path = os.path.join(Config.INPUT_ROOT, rel_path)

        try:
            # Use soundfile for fast reading
            audio, sr = sf.read(full_path)

            # Convert to torch tensor
            waveform = torch.from_numpy(audio).float()

            # Handle channels (mean if stereo)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=1)

            # 1. Duration
            duration = len(waveform) / sr

            # 2. RMS (Root Mean Square)
            rms = torch.sqrt(torch.mean(waveform**2)).item()

            # 3. Spectral Centroid
            # Needs (Channel, Time) or (Batch, Channel, Time)
            if len(waveform) > 512:
                # Add batch dim: (1, Time)
                cent_ts = spec_centroid_fn(waveform.unsqueeze(0))
                centroid = cent_ts.mean().item()
            else:
                centroid = 0.0

            # 4. Zero Crossing Rate
            zcr = (
                torch.sum(torch.abs(torch.diff(torch.sign(waveform))))
                / (2 * len(waveform))
            ).item()

            feature_records.append(
                {
                    "error": error_magnitudes[i],
                    "duration": duration,
                    "rms": rms,
                    "spectral_centroid": centroid,
                    "zero_crossing_rate": zcr,
                }
            )

        except Exception as e:
            # Skip corrupted files or read errors
            continue

    df_analysis = pd.DataFrame(feature_records)

    # Calculate and Print Correlations
    print("\nCorrelation between Error Magnitude and Input Features:")
    feature_cols = ["duration", "rms", "spectral_centroid", "zero_crossing_rate"]

    for col in feature_cols:
        if col in df_analysis.columns:
            # Pearson correlation
            corr = df_analysis["error"].corr(df_analysis[col])
            print(f"  {col}: {corr:.4f}")

    # 6. Submission Logic
    # Threshold from requirements
    THRESHOLD = 0.9872909698996656

    if final_acc > THRESHOLD:
        print(
            f"\nValidation metric ({final_acc:.8f}) exceeds threshold ({THRESHOLD:.8f})."
        )

        # The Trainer generated submission at Config.WORK_DIR/submission.csv
        src_path = os.path.join(Config.WORK_DIR, "submission.csv")
        dst_path = os.path.join(submission_dir, "submission.csv")

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"Submission file copied to: {dst_path}")
        else:
            print(
                "Error: Source submission file not found despite training completion."
            )
    else:
        print(
            f"\nValidation metric ({final_acc:.8f}) does NOT meet threshold ({THRESHOLD:.8f})."
        )
        print("Submission file will not be generated in the output folder.")


if __name__ == "__main__":
    main()
