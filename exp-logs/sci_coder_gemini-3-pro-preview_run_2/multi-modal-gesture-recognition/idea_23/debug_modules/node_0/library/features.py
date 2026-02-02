import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torchaudio.transforms as T
from library.config import Config
from library.utils import generate_gaussian_boundary_targets

# Gesture Vocabulary Mapping
GESTURE_MAP = {
    "vattene": 1,
    "vieniqui": 2,
    "perfetto": 3,
    "furbo": 4,
    "cheduepalle": 5,
    "chevuoi": 6,
    "daccordo": 7,
    "seipazzo": 8,
    "combinato": 9,
    "freganiente": 10,
    "ok": 11,
    "cosatifarei": 12,
    "basta": 13,
    "prendere": 14,
    "noncenepiu": 15,
    "fame": 16,
    "tantotempo": 17,
    "buonissimo": 18,
    "messidaccordo": 19,
    "sonostufo": 20,
}


def load_mat_file(path):
    """Safely loads a .mat file."""
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def extract_skeleton_features(video_struct, num_frames):
    """
    Extracts and normalizes skeleton features (Position, Velocity, Bone Vectors).
    Returns a numpy array of shape (NumFrames, FeatureDim).
    """
    # 1. Extract Raw Skeleton Data
    # The structure is Video.Frames[i].Skeleton.WorldPosition
    # We need to handle cases where Frames might be missing or structured differently
    frames = getattr(video_struct, "Frames", [])

    # Pre-allocate array: (NumFrames, Total_Joints, 3)
    # Total joints in Kinect v2 is usually 20 or 25. The dataset desc says 20.
    # We will dynamically determine based on the first frame or default to 20.
    raw_skeletons = np.zeros((num_frames, 20, 3), dtype=np.float32)

    if isinstance(frames, np.ndarray) and len(frames) > 0:
        actual_frames = min(len(frames), num_frames)
        for i in range(actual_frames):
            try:
                skel = frames[i].Skeleton
                # WorldPosition might be an object with X, Y, Z attributes or a struct
                # Based on desc: "WorldPosition... X, Y, Z"
                # If squeeze_me=True, it might be an object

                # Check if JointsType is available to map indices, otherwise assume standard order
                # Standard Kinect order usually matches the list in description

                # We iterate through the joints.
                # Note: The .mat structure for Skeleton usually contains an array of joints
                # or a struct with fields. Let's assume it's an array of structs or a struct of arrays.
                # Given description: "Skeleton ... JointsType ... WorldPosition"

                # If Skeleton is an array of joint objects:
                if isinstance(skel, np.ndarray):
                    for j_idx, joint in enumerate(skel):
                        if j_idx < 20:
                            pos = joint.WorldPosition
                            raw_skeletons[i, j_idx] = [pos.X, pos.Y, pos.Z]
                # If Skeleton is a single object (unlikely for multiple joints) or other format
                # We will try to handle the specific format implied by standard datasets
                else:
                    # Fallback: maybe Skeleton has a field 'WorldPosition' which is 20x3?
                    # Or Skeleton is a struct with fields like 'HipCenter', etc.
                    # The prompt says: "Skeleton Frame: An array of Skeleton structures... JointsType..."
                    # Actually, usually 'Frames' is an array of structs. 'Frames[i]' is one frame.
                    # 'Frames[i].Skeleton' might be the skeleton data.
                    pass
            except Exception:
                pass

    # 2. Select Joints
    # Shape: (NumFrames, 12, 3)
    selected_indices = Config.SELECTED_JOINTS
    skeletons = raw_skeletons[:, selected_indices, :]

    # 3. Normalize
    # Center around HipCenter (Index 0 in SELECTED_JOINTS)
    # Scale from mm to meters
    hip_center = skeletons[:, 0:1, :]  # (NumFrames, 1, 3)
    skeletons_centered = (skeletons - hip_center) / 1000.0

    # 4. Velocities
    # (NumFrames, 12, 3)
    # v[t] = p[t] - p[t-1]
    velocities = np.zeros_like(skeletons_centered)
    velocities[1:] = skeletons_centered[1:] - skeletons_centered[:-1]

    # 5. Bone Vectors
    # Shape: (NumFrames, 11, 3)
    bones = []
    for p_idx, c_idx in Config.BONE_PAIRS:
        # Map global indices to selected indices
        # We need to find where p_idx and c_idx are in SELECTED_JOINTS
        try:
            p_local = selected_indices.index(p_idx)
            c_local = selected_indices.index(c_idx)
            bone_vec = (
                skeletons_centered[:, c_local, :] - skeletons_centered[:, p_local, :]
            )
            bones.append(bone_vec)
        except ValueError:
            # If bone indices not in selected joints, skip or zero
            bones.append(np.zeros((num_frames, 3), dtype=np.float32))

    bones_arr = np.stack(bones, axis=1)  # (NumFrames, 11, 3)

    # Flatten features
    # Pos: 12*3 = 36
    # Vel: 12*3 = 36
    # Bones: 11*3 = 33
    feat_pos = skeletons_centered.reshape(num_frames, -1)
    feat_vel = velocities.reshape(num_frames, -1)
    feat_bones = bones_arr.reshape(num_frames, -1)

    return np.concatenate([feat_pos, feat_vel, feat_bones], axis=1)


def extract_audio_features(audio_path, target_num_frames):
    """
    Extracts MFCC features and aligns them to the video frame count.
    Returns numpy array of shape (NumFrames, N_MFCC).
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != Config.AUDIO_SR:
            resampler = T.Resample(sample_rate, Config.AUDIO_SR)
            waveform = resampler(waveform)

        # Compute MFCC
        mfcc_transform = T.MFCC(
            sample_rate=Config.AUDIO_SR,
            n_mfcc=Config.N_MFCC,
            melkwargs={
                "n_fft": Config.N_FFT,
                "hop_length": Config.HOP_LENGTH,
                "center": True,
            },
        )
        mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)

        # Average over channels if stereo
        if mfcc.shape[0] > 1:
            mfcc = torch.mean(mfcc, dim=0, keepdim=True)

        # Align to video frames using interpolation
        # Input to interpolate must be (Batch, Channels, Time)
        # We treat n_mfcc as channels for interpolation
        mfcc = mfcc.unsqueeze(
            0
        )  # (1, 1, n_mfcc, time) -> actually we want (1, n_mfcc, time)
        mfcc = mfcc.squeeze(1)  # (1, n_mfcc, time)

        mfcc_interpolated = torch.nn.functional.interpolate(
            mfcc, size=target_num_frames, mode="linear", align_corners=False
        )

        # Shape: (1, n_mfcc, target_num_frames) -> (target_num_frames, n_mfcc)
        mfcc_out = mfcc_interpolated.squeeze(0).permute(1, 0).numpy()
        return mfcc_out

    except Exception as e:
        # Return zeros if audio fails
        return np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)


def process_labels(video_struct, num_frames):
    """
    Parses label structure and generates frame-wise labels.
    """
    labels = np.zeros(num_frames, dtype=np.int64)

    raw_labels = getattr(video_struct, "Labels", [])

    # Helper to process a single label object
    def apply_label(obj):
        try:
            name = obj.Name
            start = int(obj.Begin) - 1  # 1-based to 0-based
            end = int(obj.End)  # inclusive in matlab usually, so end index is End

            # Clip to valid range
            start = max(0, start)
            end = min(num_frames, end)

            if name in GESTURE_MAP:
                gid = GESTURE_MAP[name]
                labels[start:end] = gid
        except AttributeError:
            pass

    if isinstance(raw_labels, np.ndarray):
        if raw_labels.ndim == 0:
            apply_label(raw_labels.item())
        else:
            for l in raw_labels:
                apply_label(l)
    else:
        apply_label(raw_labels)

    return labels


def process_sample(row, is_test=False):
    """
    Processes a single sample: loads data, extracts features, generates targets.
    """
    sample_id = row["sample_id"]
    data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
    audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

    # Load MAT file
    mat = load_mat_file(data_path)
    if mat is None or not hasattr(mat, "Video"):
        return None

    video = mat.Video
    num_frames = int(getattr(video, "NumFrames", 0))
    if num_frames == 0:
        return None

    # 1. Visual Features
    visual_feats = extract_skeleton_features(video, num_frames)

    # 2. Audio Features
    audio_feats = extract_audio_features(audio_path, num_frames)

    # Concatenate: (NumFrames, 118)
    features = np.concatenate([visual_feats, audio_feats], axis=1)

    # 3. Targets
    if not is_test:
        frame_labels = process_labels(video, num_frames)
        boundary_targets = generate_gaussian_boundary_targets(frame_labels, num_frames)
    else:
        frame_labels = np.zeros(num_frames, dtype=np.int64)
        boundary_targets = np.zeros(num_frames, dtype=np.float32)

    return {
        "features": features.astype(np.float32),
        "labels": frame_labels.astype(np.int64),
        "boundaries": boundary_targets.astype(np.float32),
        "sample_id": str(sample_id),
    }


def prepare_datasets(load_cached_data=True):
    """
    Main function to prepare training, validation, and test datasets.
    Handles caching to .npz files.
    """
    # Define cache paths
    paths = {
        "train": (Config.TRAIN_METADATA_PATH, Config.TRAIN_CACHE_PATH, False),
        "val": (Config.VAL_METADATA_PATH, Config.VAL_CACHE_PATH, False),
        "test": (Config.TEST_METADATA_PATH, Config.TEST_CACHE_PATH, True),
    }

    results = {}

    for split, (meta_path, cache_path, is_test) in paths.items():
        # Try loading cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split} data from cache: {cache_path}")
            try:
                data = np.load(cache_path, allow_pickle=True)
                results[split] = {
                    "features": list(data["features"]),
                    "labels": list(data["labels"]),
                    "boundaries": list(data["boundaries"]),
                    "sample_ids": list(data["sample_ids"]),
                }
                continue
            except Exception as e:
                print(f"Failed to load cache for {split}: {e}. Recomputing...")

        # Compute from scratch
        print(f"Processing {split} data...")
        df = pd.read_csv(meta_path)

        # Debug subset
        if Config.DEBUG_SUBSET_SIZE is not None:
            df = df.head(Config.DEBUG_SUBSET_SIZE)

        features_list = []
        labels_list = []
        boundaries_list = []
        ids_list = []

        for _, row in df.iterrows():
            sample_data = process_sample(row, is_test=is_test)
            if sample_data is not None:
                features_list.append(sample_data["features"])
                labels_list.append(sample_data["labels"])
                boundaries_list.append(sample_data["boundaries"])
                ids_list.append(sample_data["sample_id"])

        # Save to cache using object array for variable length sequences
        # Note: np.savez handles arrays. For lists of arrays of different lengths,
        # we must use allow_pickle=True when loading, and save as object array.
        feat_arr = np.array(features_list, dtype=object)
        lbl_arr = np.array(labels_list, dtype=object)
        bnd_arr = np.array(boundaries_list, dtype=object)
        id_arr = np.array(ids_list, dtype=object)

        np.savez_compressed(
            cache_path,
            features=feat_arr,
            labels=lbl_arr,
            boundaries=bnd_arr,
            sample_ids=id_arr,
        )

        results[split] = {
            "features": features_list,
            "labels": labels_list,
            "boundaries": boundaries_list,
            "sample_ids": ids_list,
        }

    return results
