import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torchaudio
import joblib
import cv2
from library.config import (
    INPUT_DIR,
    WORK_DIR,
    MFCC_COEFFS,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    IMG_SIZE,
    SEED,
    GLOBAL_MAX_SAMPLE_SIZE,
)

# Set fixed seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# Constants derived from data analysis
SAMPLE_RATE = 100  # 60,000 samples / 600 seconds (10 mins)
NYQUIST = SAMPLE_RATE / 2
# Adjust FMAX if it exceeds Nyquist (Config has 20000, which is > 50)
EFFECTIVE_FMAX = FMAX if FMAX < NYQUIST else float(NYQUIST)


def extract_time_domain_features(x: np.ndarray) -> dict:
    """
    Extracts Full Statistics and Energy features from a time-series signal.
    Preserves magnitude information (Min/Max) as per the Multi-View Policy.
    """
    features = {}

    # 1. Full Statistics (Magnitude-Preserving)
    features["mean"] = np.mean(x)
    features["std"] = np.std(x)
    features["skew"] = float(stats.skew(x))
    features["kurtosis"] = float(stats.kurtosis(x))
    features["min"] = np.min(x)
    features["max"] = np.max(x)

    # Quantiles
    q = np.quantile(x, [0.01, 0.05, 0.95, 0.99])
    features["q01"] = q[0]
    features["q05"] = q[1]
    features["q95"] = q[2]
    features["q99"] = q[3]

    # 2. Energy Features (Absolute Quantiles)
    abs_x = np.abs(x)
    features["abs_mean"] = np.mean(abs_x)
    features["abs_std"] = np.std(abs_x)
    features["abs_max"] = np.max(abs_x)

    q_abs = np.quantile(abs_x, [0.01, 0.05, 0.95, 0.99])
    features["abs_q01"] = q_abs[0]
    features["abs_q05"] = q_abs[1]
    features["abs_q95"] = q_abs[2]
    features["abs_q99"] = q_abs[3]

    # 3. Structural Features: Raw Zero-Crossing Rate
    # Using offset as amplitude gate implicitly
    zcr = ((x[:-1] * x[1:]) < 0).sum()
    features["zcr"] = zcr

    return features


def extract_freq_domain_features(x: np.ndarray) -> dict:
    """
    Extracts Robust Statistics from Frequency Domain.
    Drops Min/Max to filter transient spectral artifacts.
    """
    features = {}

    # FFT
    fft_vals = np.abs(np.fft.rfft(x))
    features["fft_mean"] = np.mean(fft_vals)
    features["fft_std"] = np.std(fft_vals)
    features["fft_q05"] = np.quantile(fft_vals, 0.05)
    features["fft_q95"] = np.quantile(fft_vals, 0.95)
    features["fft_dom_freq"] = np.argmax(fft_vals)  # Index of dominant frequency

    # MFCC
    # Convert to tensor for torchaudio
    x_tensor = torch.from_numpy(x).float().unsqueeze(0)  # (1, Time)

    # Extract MFCCs (Coeffs 1-13). We ask for 14 and drop the 0th (energy/DC)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=MFCC_COEFFS + 1,
        melkwargs={
            "n_fft": N_FFT,
            "n_mels": N_MELS,
            "hop_length": HOP_LENGTH,
            "f_min": FMIN,
            "f_max": EFFECTIVE_FMAX,
        },
    )

    try:
        mfcc = mfcc_transform(x_tensor).squeeze(0).numpy()  # (n_mfcc, frames)
        # Drop 0th coefficient
        mfcc = mfcc[1:, :]

        # Compute stats for each coefficient
        for i in range(mfcc.shape[0]):
            coeff_vals = mfcc[i, :]
            features[f"mfcc_{i+1}_mean"] = np.mean(coeff_vals)
            features[f"mfcc_{i+1}_std"] = np.std(coeff_vals)
            features[f"mfcc_{i+1}_q05"] = np.quantile(coeff_vals, 0.05)
            features[f"mfcc_{i+1}_q95"] = np.quantile(coeff_vals, 0.95)

    except Exception:
        # Fallback if signal is too short or silent
        for i in range(MFCC_COEFFS):
            features[f"mfcc_{i+1}_mean"] = 0.0
            features[f"mfcc_{i+1}_std"] = 0.0
            features[f"mfcc_{i+1}_q05"] = 0.0
            features[f"mfcc_{i+1}_q95"] = 0.0

    return features


def process_segment_tabular(segment_id: int, file_path: str) -> dict:
    """
    Worker function to process a single CSV file for tabular features.
    """
    full_path = os.path.join(INPUT_DIR, file_path)

    if not os.path.exists(full_path):
        return None

    try:
        # Load data (float32 to handle NaNs and memory)
        df = pd.read_csv(full_path, dtype="float32")
        df.fillna(0, inplace=True)  # Fill NaNs with 0 (equilibrium)

        row_features = {"segment_id": segment_id}

        # Per-Sensor Features
        for col in df.columns:
            if not col.startswith("sensor_"):
                continue

            x = df[col].values

            # Time Domain
            t_feats = extract_time_domain_features(x)
            for k, v in t_feats.items():
                row_features[f"{col}_{k}"] = v

            # Freq Domain
            f_feats = extract_freq_domain_features(x)
            for k, v in f_feats.items():
                row_features[f"{col}_{k}"] = v

        # Spatial Features: Correlation Matrix
        # Flatten upper triangle of correlation matrix
        corr_matrix = df.corr().abs()
        sensors = [c for c in df.columns if c.startswith("sensor_")]
        for i in range(len(sensors)):
            for j in range(i + 1, len(sensors)):
                s1, s2 = sensors[i], sensors[j]
                row_features[f"corr_{s1}_{s2}"] = corr_matrix.loc[s1, s2]

        return row_features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def generate_tabular_features(
    metadata_path: str, cache_path: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Orchestrates the parallel extraction of tabular features with caching.
    """
    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached tabular features from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Generating tabular features from {metadata_path}...")

    # 2. Load Metadata
    df_meta = pd.read_csv(metadata_path)

    # 3. Parallel Processing
    # Use joblib to parallelize over files
    results = joblib.Parallel(n_jobs=12, backend="loky")(
        joblib.delayed(process_segment_tabular)(row["segment_id"], row["file_path"])
        for _, row in df_meta.iterrows()
    )

    # Filter None results
    results = [r for r in results if r is not None]

    # 4. Create DataFrame
    df_features = pd.DataFrame(results)

    # Merge with targets if available (train/val)
    if "time_to_eruption" in df_meta.columns:
        df_features = df_features.merge(
            df_meta[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
        )

    # 5. Save Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_features.to_parquet(cache_path, index=False)
    print(f"Saved tabular features to {cache_path}")

    return df_features


def generate_mel_spectrogram(df: pd.DataFrame) -> np.ndarray:
    """
    Converts a sensor DataFrame to a Multi-Channel Log-Mel Spectrogram.
    Output Shape: (10, 224, 224)
    """
    # Prepare Tensor: (Channels, Time)
    # Ensure all 10 sensors are present and in order
    sensors = [f"sensor_{i}" for i in range(1, 11)]
    data = []
    for s in sensors:
        if s in df.columns:
            data.append(df[s].values)
        else:
            data.append(np.zeros(len(df), dtype=np.float32))

    x_tensor = torch.tensor(np.array(data), dtype=torch.float32)  # (10, 60001)

    # Mel Spectrogram Transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=FMIN,
        f_max=EFFECTIVE_FMAX,
        power=2.0,
    )

    # Compute Spectrogram
    # Output: (10, n_mels, time_frames)
    mels = mel_transform(x_tensor)

    # Convert to Log Scale (Log-Mel)
    # log(S + 1e-9) to avoid log(0).
    # Note: The strategy asks for log(X+1) later for normalization,
    # but standard Log-Mel is usually log(S).
    # We will return the linear magnitude here or standard log-mel?
    # The prompt strategy says: "Convert ... to Log-Mel Spectrograms"
    # AND "Normalize ... X_norm = log(X+1)/log(M+1)".
    # This implies X is the LINEAR Mel Spectrogram.
    # So we return LINEAR Mel Spectrogram here.

    # Resize to (224, 224)
    # mels shape: (10, 224, ~235)
    mels_np = mels.numpy()
    resized_channels = []

    for c in range(mels_np.shape[0]):
        # Resize using OpenCV (Width, Height) -> (224, 224)
        # Input to resize is (Height, Width) if using src, but cv2 expects (Width, Height) for dsize
        img = mels_np[c]  # (224, ~235)
        # Resize to IMG_SIZE (224, 224)
        # cv2.resize(src, dsize=(width, height))
        res = cv2.resize(
            img, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR
        )
        resized_channels.append(res)

    spectrogram = np.stack(resized_channels, axis=0)  # (10, 224, 224)
    return spectrogram


def compute_global_max(
    metadata_path: str,
    cache_path: str,
    sample_size: int = GLOBAL_MAX_SAMPLE_SIZE,
    load_cached_data: bool = True,
) -> float:
    """
    Computes the global maximum value M from a subset of training data.
    Used for Global Log-Max Scaling.
    """
    if load_cached_data and os.path.exists(cache_path):
        val = np.load(cache_path).item()
        print(f"Loaded Global Max: {val}")
        return val

    print("Computing Global Max for Normalization...")
    df_meta = pd.read_csv(metadata_path)

    # Sample subset
    if len(df_meta) > sample_size:
        df_sample = df_meta.sample(n=sample_size, random_state=SEED)
    else:
        df_sample = df_meta

    max_val = 0.0

    def process_max(file_path):
        full_path = os.path.join(INPUT_DIR, file_path)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path, dtype="float32").fillna(0)
            spec = generate_mel_spectrogram(df)
            return np.max(spec)
        return 0.0

    # Parallel execution
    results = joblib.Parallel(n_jobs=12, backend="loky")(
        joblib.delayed(process_max)(row["file_path"]) for _, row in df_sample.iterrows()
    )

    if results:
        max_val = max(results)

    # Save
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, np.array(max_val))
    print(f"Computed and Saved Global Max: {max_val}")

    return max_val


def generate_vision_dataset(
    metadata_path: str,
    output_dir: str,
    global_max: float,
    load_cached_data: bool = True,
):
    """
    Generates and saves normalized spectrograms for the Vision Branch.
    Normalization: X_norm = log(X + 1) / log(M + 1)
    """
    # Check if output dir is populated
    os.makedirs(output_dir, exist_ok=True)

    df_meta = pd.read_csv(metadata_path)

    # Check if already processed (simple count check)
    existing_files = os.listdir(output_dir)
    if load_cached_data and len(existing_files) >= len(df_meta):
        print(f"Vision dataset at {output_dir} appears complete. Skipping generation.")
        return

    print(f"Generating vision dataset in {output_dir}...")

    log_m_plus_1 = np.log(global_max + 1.0)

    def process_and_save(segment_id, file_path):
        save_path = os.path.join(output_dir, f"{segment_id}.npy")
        if load_cached_data and os.path.exists(save_path):
            return

        full_path = os.path.join(INPUT_DIR, file_path)
        if not os.path.exists(full_path):
            return

        df = pd.read_csv(full_path, dtype="float32").fillna(0)

        # 1. Generate Linear Spectrogram
        spec = generate_mel_spectrogram(df)  # (10, 224, 224)

        # 2. Global Log-Max Scaling
        # X_norm = log(X + 1) / log(M + 1)
        spec_norm = np.log(spec + 1.0) / log_m_plus_1

        # 3. Clip to [0, 1] just in case of outliers > global_max
        spec_norm = np.clip(spec_norm, 0.0, 1.0)

        # 4. Save as float32
        np.save(save_path, spec_norm.astype(np.float32))

    joblib.Parallel(n_jobs=12, backend="loky")(
        joblib.delayed(process_and_save)(row["segment_id"], row["file_path"])
        for _, row in df_meta.iterrows()
    )

    print(f"Vision dataset generation complete.")
