import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import scipy.stats
from concurrent.futures import ProcessPoolExecutor
from library import config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(config.SEED)


def compute_beamformed_signal(df_segment):
    """
    Computes a 'Virtual Source' channel by averaging normalized waveforms of all 10 sensors.
    This acts as a beamformer, constructively interfering the common seismic signal.
    """
    # Select physical sensors
    sensor_cols = [c for c in df_segment.columns if "sensor_" in c]

    # Normalize each sensor (Z-score) to ensure equal contribution
    # Handle potentially constant signals (std=0) by adding epsilon
    data = df_segment[sensor_cols].values
    means = np.nanmean(data, axis=0)
    stds = np.nanstd(data, axis=0)
    stds[stds == 0] = 1.0  # Avoid division by zero

    normalized_data = (data - means) / stds

    # Beamforming: Average across sensors
    virtual_signal = np.nanmean(normalized_data, axis=1)

    return pd.Series(virtual_signal, name="virtual_source")


def get_mfcc_transform(sample_rate):
    """Factory for MFCC transform based on config."""
    return T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=config.MFCC_PARAMS["n_mfcc"],
        melkwargs={
            "n_fft": config.MFCC_PARAMS["n_fft"],
            "n_mels": 128,  # Standard internal mel bin count before MFCC
            "hop_length": config.MFCC_PARAMS["hop_length"],
            "mel_scale": "htk",
        },
    )


def get_spectrogram_transform(sample_rate):
    """Factory for Log-Mel Spectrogram transform based on config."""
    # Pipeline: MelSpectrogram -> AmplitudeToDB -> Resize
    mel_spec = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=config.SPECTROGRAM_PARAMS["n_fft"],
        hop_length=config.SPECTROGRAM_PARAMS["hop_length"],
        n_mels=config.SPECTROGRAM_PARAMS["n_mels"],
        f_min=config.SPECTROGRAM_PARAMS["fmin"],
        f_max=config.SPECTROGRAM_PARAMS["fmax"],
    )
    return mel_spec


def extract_tabular_features(df_segment, virtual_signal=None):
    """
    Extracts parsimonious statistical and spectral features from the segment.
    Includes physical sensors and the optional virtual beamformed sensor.
    """
    features = {}

    # Prepare list of signals to process
    signals = {
        col: df_segment[col].values for col in df_segment.columns if "sensor_" in col
    }
    if virtual_signal is not None:
        signals["virtual_source"] = virtual_signal.values

    # Initialize MFCC transform (CPU is fine for feature extraction loop)
    # We assume a nominal sampling rate. 60001 samples / 600 seconds = ~100 Hz
    sr = config.MFCC_PARAMS["sr"]
    mfcc_transform = get_mfcc_transform(sr)

    for sensor_name, signal in signals.items():
        # Handle NaNs: Fill with mean or 0
        if np.isnan(signal).any():
            signal = np.nan_to_num(signal, nan=np.nanmean(signal))

        # --- 1. Global & Absolute Statistics ---
        features[f"{sensor_name}_mean"] = np.mean(signal)
        features[f"{sensor_name}_std"] = np.std(signal)
        features[f"{sensor_name}_skew"] = scipy.stats.skew(signal)
        features[f"{sensor_name}_kurtosis"] = scipy.stats.kurtosis(signal)

        # Quantiles
        q_vals = np.percentile(signal, [1, 5, 95, 99])
        features[f"{sensor_name}_q01"] = q_vals[0]
        features[f"{sensor_name}_q05"] = q_vals[1]
        features[f"{sensor_name}_q95"] = q_vals[2]
        features[f"{sensor_name}_q99"] = q_vals[3]

        # Absolute Quantiles (Energy)
        abs_signal = np.abs(signal)
        abs_q_vals = np.percentile(abs_signal, [95, 99])
        features[f"{sensor_name}_abs_q95"] = abs_q_vals[0]
        features[f"{sensor_name}_abs_q99"] = abs_q_vals[1]

        # --- 2. Structural Features ---
        # Raw Zero-Crossing Rate (without centering)
        # Counts how often the signal crosses 0 (polarity shift)
        zcr = ((signal[:-1] * signal[1:]) < 0).sum()
        features[f"{sensor_name}_raw_zcr"] = zcr

        # --- 3. Parsimonious MFCCs ---
        # Convert to tensor for torchaudio
        sig_tensor = torch.from_numpy(signal).float().unsqueeze(0)  # (1, time)
        mfcc = mfcc_transform(sig_tensor).squeeze(0).numpy()  # (n_mfcc, time_frames)

        # Aggregate over time using ONLY Robust Statistics
        # MFCCs are dim 0. We compute stats across dim 1 (time)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_q05 = np.percentile(mfcc, 5, axis=1)
        mfcc_q95 = np.percentile(mfcc, 95, axis=1)

        for i in range(config.MFCC_PARAMS["n_mfcc"]):
            # Coefficient 0 is often energy, 1-12 are shape. We keep all 13 configured.
            features[f"{sensor_name}_mfcc_{i}_mean"] = mfcc_mean[i]
            features[f"{sensor_name}_mfcc_{i}_std"] = mfcc_std[i]
            features[f"{sensor_name}_mfcc_{i}_q05"] = mfcc_q05[i]
            features[f"{sensor_name}_mfcc_{i}_q95"] = mfcc_q95[i]

    # --- 4. Spatial Features ---
    # Correlation matrix between physical sensors
    # We only care about the upper triangle of the correlation matrix of physical sensors
    phys_sensors = [c for c in df_segment.columns if "sensor_" in c]
    if len(phys_sensors) > 1:
        corr_matrix = df_segment[phys_sensors].corr().values
        # Extract upper triangle indices
        triu_indices = np.triu_indices(len(phys_sensors), k=1)
        corr_values = corr_matrix[triu_indices]

        # Store individual pair correlations or aggregates?
        # Given parsimony, let's store basic stats of correlations and top pairs if needed.
        # However, "Spatial Features: Compute Correlation Matrix" implies using the values.
        # To avoid feature explosion (45 pairs), we can store them, but let's stick to the plan.
        # "Compute the Correlation Matrix" -> usually implies flattening it into features.
        k = 0
        for i in range(len(phys_sensors)):
            for j in range(i + 1, len(phys_sensors)):
                features[f"corr_{phys_sensors[i]}_{phys_sensors[j]}"] = corr_matrix[
                    i, j
                ]

    return features


def create_spectrograms(df_segment):
    """
    Generates a stacked Log-Mel Spectrogram tensor for the vision branch.
    Output shape: (10, H, W) where H, W are defined in config.
    """
    phys_sensors = [f"sensor_{i}" for i in range(1, 11)]

    # Configuration
    sr = config.MFCC_PARAMS["sr"]
    target_size = config.CNN_CONFIG["img_size"]  # (128, 128)

    # Transforms
    mel_transform = get_spectrogram_transform(sr)
    amp_to_db = T.AmplitudeToDB()

    # We need to resize to fixed dimensions.
    # MelSpectrogram outputs (n_mels, time). We want (128, 128).
    # Since n_mels is 128, we just need to resize the time dimension or both.
    resize_transform = T.Resize(target_size)

    specs = []

    for sensor in phys_sensors:
        if sensor in df_segment.columns:
            sig = df_segment[sensor].values
            # Fill NaNs
            if np.isnan(sig).any():
                sig = np.nan_to_num(sig, nan=0.0)

            sig_t = torch.from_numpy(sig).float()

            # Compute Mel Spec
            spec = mel_transform(sig_t)  # (n_mels, time)

            # Log Scale
            spec = amp_to_db(spec)

            # Resize. Resize expects (C, H, W) or (H, W).
            # spec is (H, W). Add channel dim for resize then remove.
            spec = spec.unsqueeze(0)
            spec = resize_transform(spec)
            spec = spec.squeeze(0)

            specs.append(spec.numpy())
        else:
            # Fallback for missing sensor columns (unlikely based on data desc)
            specs.append(np.zeros(target_size, dtype=np.float32))

    # Stack to (10, 128, 128)
    return np.stack(specs, axis=0)


def process_single_file(args):
    """
    Worker function for parallel processing.
    args: (file_rel_path, segment_id, use_beamforming)
    """
    file_rel_path, segment_id, use_beamforming = args
    full_path = os.path.join(config.INPUT_DIR, file_rel_path)

    if not os.path.exists(full_path):
        # Should not happen given metadata verification, but safe fallback
        return None, None, None

    try:
        # Load CSV
        df = pd.read_csv(full_path, dtype="float32")

        # Beamforming
        virtual_sig = None
        if use_beamforming:
            virtual_sig = compute_beamformed_signal(df)

        # Tabular Features
        tab_feats = extract_tabular_features(df, virtual_sig)
        tab_feats["segment_id"] = segment_id

        # Spectrograms
        spec_tensor = create_spectrograms(df)

        return tab_feats, spec_tensor, segment_id

    except Exception as e:
        print(f"Error processing {file_rel_path}: {e}")
        return None, None, None


def load_and_process_data(
    metadata_path, output_tabular_path, output_spectrogram_path, load_cached_data=True
):
    """
    Main driver for data loading and feature engineering.
    Handles caching, parallel processing, and data alignment.
    """
    # 1. Check Cache
    if (
        load_cached_data
        and os.path.exists(output_tabular_path)
        and os.path.exists(output_spectrogram_path)
    ):
        print(
            f"Loading cached features from {output_tabular_path} and {output_spectrogram_path}"
        )
        df_features = pd.read_parquet(output_tabular_path)
        spectrograms = np.load(output_spectrogram_path)

        # Load targets from metadata to ensure alignment
        df_meta = pd.read_csv(metadata_path)
        # Merge targets onto features
        if "time_to_eruption" in df_meta.columns:
            # Ensure order matches
            df_features = (
                df_features.set_index("segment_id")
                .loc[df_meta["segment_id"]]
                .reset_index()
            )
            targets = (
                df_meta.set_index("segment_id")
                .loc[df_features["segment_id"]]["time_to_eruption"]
                .values
            )
        else:
            targets = None

        return df_features, spectrograms, targets

    # 2. Process from Scratch
    print(f"Processing data from {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)

    # Prepare arguments for parallel worker
    tasks = []
    for _, row in df_meta.iterrows():
        tasks.append((row["file_path"], row["segment_id"], config.USE_BEAMFORMING))

    # Execute in parallel
    # Note: torchaudio might have issues with multiprocessing if not handled carefully,
    # but since we create transforms inside functions (or they are pure), it should be fine.
    # Limiting workers to avoid OOM or CPU contention.
    tabular_results = []
    spectrogram_results = []
    valid_ids = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_single_file, tasks))

    for tab, spec, seg_id in results:
        if tab is not None:
            tabular_results.append(tab)
            spectrogram_results.append(spec)
            valid_ids.append(seg_id)

    # Aggregate Results
    df_features = pd.DataFrame(tabular_results)
    spectrograms = np.stack(spectrogram_results, axis=0)  # (N, 10, 128, 128)

    # Align targets
    if "time_to_eruption" in df_meta.columns:
        # Create a mapping from ID to target
        target_map = dict(zip(df_meta["segment_id"], df_meta["time_to_eruption"]))
        targets = np.array([target_map[sid] for sid in valid_ids])
    else:
        targets = None  # Test set

    # 3. Save to Cache
    print(f"Saving processed features to {output_tabular_path}...")
    df_features.to_parquet(output_tabular_path, index=False)

    print(f"Saving spectrograms to {output_spectrogram_path}...")
    np.save(output_spectrogram_path, spectrograms)

    return df_features, spectrograms, targets
