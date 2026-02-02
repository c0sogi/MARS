import os
import numpy as np
import scipy.io
import torch
import torchaudio
import soundfile as sf
from library.config import (
    AUDIO_SAMPLE_RATE,
    N_MFCC,
    NUM_JOINTS,
    JOINTS_DIM,
    WORKING_DIR,
)

# Define cache directory based on WORKING_DIR
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def load_robust_mat(mat_path, load_cached_data=True):
    """
    Robustly parses .mat files to extract skeleton data.
    Handles polymorphic structures (struct array vs cell array vs object).
    Implements caching using .npy files.
    """
    # 1. Determine Cache Path
    filename = os.path.basename(mat_path)
    sample_id = filename.split("_")[0]
    cache_path = os.path.join(CACHE_DIR, f"{sample_id}_skeleton.npy")

    # 2. Try Load from Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            return data
        except Exception:
            # If load fails, proceed to re-compute
            pass

    # 3. Process Data
    try:
        # Load mat file, handling struct as objects
        mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    except Exception:
        # Return zeros if file is completely corrupt/missing
        # We can't determine num_frames easily without the file,
        # but usually this case is filtered out upstream.
        # Returning None signals failure.
        return None

    skeleton_data = None

    if "Video" in mat:
        video = mat["Video"]
        if hasattr(video, "Frames"):
            frames = video.Frames

            # Normalize to list/array
            if not isinstance(frames, (np.ndarray, list)):
                frames = [frames]

            num_frames = len(frames)
            skeleton_data = np.zeros(
                (num_frames, NUM_JOINTS, JOINTS_DIM), dtype=np.float32
            )

            for t, frame in enumerate(frames):
                # Defensive check for Skeleton existence
                if not hasattr(frame, "Skeleton"):
                    continue

                skel = frame.Skeleton

                # Handle case where Skeleton is empty/0
                if isinstance(skel, (int, float)) or (
                    isinstance(skel, np.ndarray) and skel.size == 0
                ):
                    continue

                # Handle array of skeletons (multi-user), take first
                if isinstance(skel, np.ndarray) or isinstance(skel, list):
                    if len(skel) > 0:
                        skel = skel[0]
                    else:
                        continue

                # Check for WorldPosition
                if not hasattr(skel, "WorldPosition"):
                    continue

                wp = skel.WorldPosition

                # Extract coordinates
                # Expecting X, Y, Z to be arrays of length 20
                try:
                    if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                        x = np.atleast_1d(wp.X)
                        y = np.atleast_1d(wp.Y)
                        z = np.atleast_1d(wp.Z)

                        if (
                            len(x) == NUM_JOINTS
                            and len(y) == NUM_JOINTS
                            and len(z) == NUM_JOINTS
                        ):
                            skeleton_data[t, :, 0] = x
                            skeleton_data[t, :, 1] = y
                            skeleton_data[t, :, 2] = z
                except Exception:
                    continue

            # Forward fill missing frames to maintain kinematic continuity
            # (Avoids massive velocity spikes from 0-padding)
            for t in range(1, num_frames):
                # If frame is all zeros (sum of abs is 0), copy previous
                if np.sum(np.abs(skeleton_data[t])) < 1e-6:
                    skeleton_data[t] = skeleton_data[t - 1]

    # If parsing failed completely, return None
    if skeleton_data is None:
        return None

    # 4. Save to Cache
    np.save(cache_path, skeleton_data)

    return skeleton_data


def compute_kinematics(positions):
    """
    Computes Velocity and Acceleration from Positions.
    Input: (T, J, 3)
    Output: (T, J, 9) -> [Pos, Vel, Acc]
    """
    # positions: T x J x 3
    # Velocity: P(t) - P(t-1)
    # Acceleration: V(t) - V(t-1)

    # Pad with first frame to maintain length T
    # We use np.gradient or simple diff. Simple diff with padding is standard for causal/sequence tasks.

    # Velocity
    # Shifted difference
    vel = np.zeros_like(positions)
    vel[1:] = positions[1:] - positions[:-1]
    # vel[0] remains 0

    # Acceleration
    acc = np.zeros_like(vel)
    acc[1:] = vel[1:] - vel[:-1]
    # acc[0] remains 0

    # Concatenate: T x J x 9
    kinematics = np.concatenate([positions, vel, acc], axis=-1)

    return kinematics


def extract_audio_mfcc(audio_path, num_frames, load_cached_data=True):
    """
    Extracts MFCC features and aligns them to the video frame count.
    Implements caching.
    """
    # 1. Determine Cache Path
    filename = os.path.basename(audio_path)
    sample_id = filename.split("_")[0]
    cache_path = os.path.join(CACHE_DIR, f"{sample_id}_audio.npy")

    # 2. Try Load from Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            # Check if cached data matches requested num_frames (simple integrity check)
            if data.shape[0] == num_frames:
                return data
            # If mismatch, recompute
        except Exception:
            pass

    # 3. Process Data
    try:
        if not os.path.exists(audio_path):
            # Return zeros if audio missing
            mfcc_aligned = np.zeros((num_frames, N_MFCC), dtype=np.float32)
        else:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if necessary
            if sample_rate != AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=AUDIO_SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Extract MFCC
            # n_mfcc=13 is standard, sample_rate=16000
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=AUDIO_SAMPLE_RATE,
                n_mfcc=N_MFCC,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )

            mfcc = mfcc_transform(waveform)  # Shape: (Channel, n_mfcc, time)

            # Average over channels if stereo
            if mfcc.shape[0] > 1:
                mfcc = torch.mean(mfcc, dim=0, keepdim=True)

            # Align to video frames using interpolation
            # Input to interpolate must be (Batch, Channels, Time)
            # Current: (1, n_mfcc, time)
            # Target time: num_frames

            if num_frames > 0:
                mfcc_aligned_tensor = torch.nn.functional.interpolate(
                    mfcc, size=num_frames, mode="linear", align_corners=False
                )
                # Shape: (1, n_mfcc, num_frames)

                # Transpose to (num_frames, n_mfcc)
                mfcc_aligned = mfcc_aligned_tensor.squeeze(0).transpose(0, 1).numpy()
            else:
                mfcc_aligned = np.zeros((0, N_MFCC), dtype=np.float32)

    except Exception as e:
        # Fallback to zeros on error
        mfcc_aligned = np.zeros((num_frames, N_MFCC), dtype=np.float32)

    # 4. Save to Cache
    np.save(cache_path, mfcc_aligned)

    return mfcc_aligned
