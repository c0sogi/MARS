import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import scipy.ndimage
import torchaudio
import warnings
from library.config import Config
from library.utils import set_seed

# Suppress warnings from librosa/scipy if any
warnings.filterwarnings("ignore")


def load_mat_data(mat_path):
    """
    Parses the .mat file to extract skeleton data and frame-wise labels.
    """
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        video = mat["Video"]
        num_frames = video.NumFrames

        # Extract Skeleton Data
        # Video.Frames is an array of structs, each having Skeleton
        frames = video.Frames

        # Pre-allocate
        # Shape: (T, NumJoints, 3)
        # We need to handle cases where Frames might be missing or singular
        if not isinstance(frames, np.ndarray):
            frames = np.array([frames])

        skeleton_data = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

        # Joint mapping from dataset indices to our selected indices
        # Dataset joints:
        # 0:HipCenter, 1:Spine, 2:ShoulderCenter, 3:Head,
        # 4:ShoulderLeft, 5:ElbowLeft, 6:WristLeft, 7:HandLeft,
        # 8:ShoulderRight, 9:ElbowRight, 10:WristRight, 11:HandRight
        # ... and others.
        # Config.SELECTED_JOINTS matches the first 12 indices of the dataset exactly.

        for t in range(min(num_frames, len(frames))):
            frame_obj = frames[t]
            if hasattr(frame_obj, "Skeleton"):
                skel = frame_obj.Skeleton
                if hasattr(skel, "WorldPosition"):
                    # WorldPosition is usually a struct array or object array
                    # We need to iterate over joints
                    # Assuming skel.WorldPosition is an array of structs for joints
                    # Or skel is an array of joints.
                    # Based on description: "Skeleton Frame: An array of Skeleton structures... contained within a Skeletons array"
                    # Actually, usually in these datasets: frame_obj.Skeleton is the list of joints.

                    # Let's try to access joints directly by index if possible, or iterate
                    # If skel is an array of joint objects:
                    joints = skel
                    if isinstance(joints, np.ndarray):
                        for j_idx, selected_j in enumerate(Config.SELECTED_JOINTS):
                            if selected_j < len(joints):
                                joint = joints[selected_j]
                                if hasattr(joint, "WorldPosition"):
                                    pos = joint.WorldPosition
                                    skeleton_data[t, j_idx, 0] = pos.X
                                    skeleton_data[t, j_idx, 1] = pos.Y
                                    skeleton_data[t, j_idx, 2] = pos.Z

        # Normalize: Hip Center Subtraction
        # HipCenter is index 0 in our selected joints (and index 0 in dataset)
        hip_pos = skeleton_data[:, 0:1, :]  # (T, 1, 3)
        skeleton_data = skeleton_data - hip_pos

        # Extract Labels
        # Initialize with Background (0)
        labels_cls = np.zeros(num_frames, dtype=np.int64)

        labels_raw = getattr(video, "Labels", [])

        # Helper to process single label entry
        def process_label_entry(l):
            try:
                name = l.Name
                start = int(l.Begin) - 1  # 1-based to 0-based
                end = int(
                    l.End
                )  # inclusive in Matlab, so end index in Python slice is End

                # Map name to ID
                # We need the GESTURE_MAP. Since it's not in Config, we recreate it or infer.
                # The prompt description lists them 1..20.
                # We'll rely on the provided metadata csv which already has the sequence of labels.
                # However, to get frame-wise, we need to know WHICH label is WHERE.
                # We must map the name string to ID.

                # Minimal map based on description
                gesture_names = [
                    "vattene",
                    "vieniqui",
                    "perfetto",
                    "furbo",
                    "cheduepalle",
                    "chevuoi",
                    "daccordo",
                    "seipazzo",
                    "combinato",
                    "freganiente",
                    "ok",
                    "cosatifarei",
                    "basta",
                    "prendere",
                    "noncenepiu",
                    "fame",
                    "tantotempo",
                    "buonissimo",
                    "messidaccordo",
                    "sonostufo",
                ]
                # Create map
                g_map = {name: i + 1 for i, name in enumerate(gesture_names)}

                if name in g_map:
                    gid = g_map[name]
                    # Clip indices
                    start = max(0, start)
                    end = min(num_frames, end)
                    labels_cls[start:end] = gid
            except AttributeError:
                pass

        if isinstance(labels_raw, np.ndarray):
            if labels_raw.ndim == 0:
                process_label_entry(labels_raw.item())
            else:
                for l in labels_raw:
                    process_label_entry(l)
        else:
            process_label_entry(labels_raw)

        return skeleton_data, labels_cls, num_frames

    except Exception as e:
        # Return zeros if failure
        # print(f"Error parsing {mat_path}: {e}")
        return None, None, 0


def load_audio_features(audio_path, target_frames):
    """
    Loads audio, computes MFCC, and interpolates to match target_frames.
    """
    try:
        # Load audio using torchaudio
        waveform, sr = torchaudio.load(audio_path)  # (Channels, Time)

        if waveform.numel() == 0:
            raise ValueError("Empty audio")

        # Resample if necessary
        if sr != Config.AUDIO_SR:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=Config.AUDIO_SR
            )
            waveform = resampler(waveform)
            sr = Config.AUDIO_SR

        # Mix down to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Compute MFCC
        # Use settings compatible with typical defaults (n_fft=2048, hop_length=512)
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=Config.N_MFCC,
            melkwargs={"n_fft": 2048, "hop_length": 512, "center": True},
        )

        mfcc = mfcc_transform(waveform)  # (1, n_mfcc, T_audio)
        mfcc = mfcc.squeeze(0)  # (n_mfcc, T_audio)
        mfcc = mfcc.numpy()  # Convert to numpy

        # Transpose to (T_audio, n_mfcc)
        mfcc = mfcc.T

        # Interpolate to match video frames
        if mfcc.shape[0] != target_frames:
            # Create time axes
            t_audio = np.linspace(0, 1, mfcc.shape[0])
            t_video = np.linspace(0, 1, target_frames)

            mfcc_resampled = np.zeros((target_frames, mfcc.shape[1]), dtype=np.float32)
            for c in range(mfcc.shape[1]):
                mfcc_resampled[:, c] = np.interp(t_video, t_audio, mfcc[:, c])
            return mfcc_resampled
        else:
            return mfcc.astype(np.float32)

    except Exception as e:
        # Return zeros
        return np.zeros((target_frames, Config.N_MFCC), dtype=np.float32)


def process_dataset(metadata_path, split_name, load_cached_data=True):
    """
    Loads metadata, processes raw files, and caches the result.
    Returns lists of: positions, audio_feats, labels_cls, labels_bnd
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{split_name}_data.npz")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {split_name} data from cache: {cache_file}")
        try:
            data = np.load(cache_file, allow_pickle=True)
            return (
                data["positions"],
                data["audio_feats"],
                data["labels_cls"],
                data["labels_bnd"],
                data["sample_ids"],
            )
        except Exception as e:
            print(f"Cache load failed: {e}. Reprocessing.")

    # 2. Process from Scratch
    print(f"Processing {split_name} data from raw files...")
    df = pd.read_csv(metadata_path)

    all_positions = []
    all_audio = []
    all_labels_cls = []
    all_labels_bnd = []
    all_sample_ids = []

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # Load Skeleton & Labels
        skel, cls_labels, num_frames = load_mat_data(mat_path)
        if skel is None:
            continue

        # Load Audio
        audio = load_audio_features(audio_path, num_frames)

        # Generate Boundary Labels
        # bnd[t] = 1 if cls[t] != cls[t-1]
        bnd_labels = np.zeros_like(cls_labels, dtype=np.float32)
        # Calculate differences
        diff = cls_labels[1:] != cls_labels[:-1]
        bnd_labels[1:][diff] = 1.0
        # First frame: usually 0 unless it starts immediately?
        # Let's keep 0.

        # For Test set, labels will be all 0 (background) from load_mat_data
        # because Video.Labels is empty/missing in test mats.
        # This is correct behavior.

        all_positions.append(skel)
        all_audio.append(audio)
        all_labels_cls.append(cls_labels)
        all_labels_bnd.append(bnd_labels)
        all_sample_ids.append(sample_id)

    # Convert to object arrays for saving (variable length)
    all_positions = np.array(all_positions, dtype=object)
    all_audio = np.array(all_audio, dtype=object)
    all_labels_cls = np.array(all_labels_cls, dtype=object)
    all_labels_bnd = np.array(all_labels_bnd, dtype=object)
    all_sample_ids = np.array(all_sample_ids, dtype=object)

    # 3. Save Cache
    Config.init_dirs()  # Ensure dir exists
    np.savez_compressed(
        cache_file,
        positions=all_positions,
        audio_feats=all_audio,
        labels_cls=all_labels_cls,
        labels_bnd=all_labels_bnd,
        sample_ids=all_sample_ids,
    )
    print(f"Saved {split_name} data to cache.")

    return all_positions, all_audio, all_labels_cls, all_labels_bnd, all_sample_ids


class GestureDataset(Dataset):
    def __init__(
        self, positions, audio_feats, labels_cls, labels_bnd, sample_ids, augment=False
    ):
        self.positions = positions
        self.audio_feats = audio_feats
        self.labels_cls = labels_cls
        self.labels_bnd = labels_bnd
        self.sample_ids = sample_ids
        self.augment = augment

    def __len__(self):
        return len(self.positions)

    def physically_consistent_augmentation(self, positions):
        """
        Applies smooth Gaussian noise to positions.
        positions: (T, J, 3)
        """
        T, J, C = positions.shape

        # Generate white noise
        noise = np.random.normal(0, Config.AUG_NOISE_SIGMA, size=(T, J, C))

        # Apply Low-Pass Filter (smoothing along time axis)
        # sigma controls the smoothness
        smooth_noise = scipy.ndimage.gaussian_filter1d(
            noise, sigma=Config.AUG_SMOOTH_SIGMA, axis=0
        )

        return positions + smooth_noise

    def __getitem__(self, idx):
        # Load raw data
        pos = self.positions[idx].astype(np.float32)  # (T, J, 3)
        audio = self.audio_feats[idx].astype(np.float32)  # (T, n_mfcc)
        cls_target = self.labels_cls[idx].astype(np.int64)  # (T,)
        bnd_target = self.labels_bnd[idx].astype(np.float32)  # (T,)

        # Augmentation
        if self.augment:
            pos = self.physically_consistent_augmentation(pos)

        # Feature Engineering: Velocity
        # V_t = P_t - P_{t-1}
        # Pad first frame with 0
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]

        # Flatten Skeleton: (T, J, 3) -> (T, J*3)
        T = pos.shape[0]
        pos_flat = pos.reshape(T, -1)
        vel_flat = vel.reshape(T, -1)

        # Concatenate: [Pos, Vel, Audio]
        # Pos: J*3, Vel: J*3, Audio: n_mfcc
        features = np.concatenate([pos_flat, vel_flat, audio], axis=1)  # (T, InputDim)

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "labels_cls": torch.tensor(cls_target, dtype=torch.long),
            "labels_bnd": torch.tensor(bnd_target, dtype=torch.float32),
            "length": T,
            "sample_id": str(self.sample_ids[idx]),
        }


def collate_fn(batch):
    """
    Pads sequences and creates masks.
    """
    # Sort by length (descending) for pack_padded_sequence if needed (optional but good practice)
    batch.sort(key=lambda x: x["length"], reverse=True)

    lengths = [x["length"] for x in batch]
    max_len = max(lengths)
    # Clamp to Config.MAX_FRAMES if needed, but usually we handle variable length
    # If using fixed size batching for TCN, we might want to pad to a multiple or fixed size.
    # Let's pad to max_len in batch.

    batch_size = len(batch)
    input_dim = batch[0]["features"].shape[1]

    padded_features = torch.zeros(batch_size, max_len, input_dim)
    padded_cls = torch.zeros(batch_size, max_len, dtype=torch.long)  # 0 is background
    padded_bnd = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float32)

    sample_ids = []

    for i, item in enumerate(batch):
        l = item["length"]
        padded_features[i, :l, :] = item["features"]
        padded_cls[i, :l] = item["labels_cls"]
        padded_bnd[i, :l] = item["labels_bnd"]
        mask[i, :l] = 1.0
        sample_ids.append(item["sample_id"])

    return {
        "features": padded_features,
        "labels_cls": padded_cls,
        "labels_bnd": padded_bnd,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "sample_ids": sample_ids,
    }


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Val, and Test.
    """
    set_seed(Config.SEED)
    Config.init_dirs()

    # Paths
    train_meta = os.path.join(Config.METADATA_DIR, "train.csv")
    val_meta = os.path.join(Config.METADATA_DIR, "val.csv")
    test_meta = os.path.join(Config.METADATA_DIR, "test.csv")

    # Load Data
    train_data = process_dataset(train_meta, "train", load_cached_data)
    val_data = process_dataset(val_meta, "val", load_cached_data)
    test_data = process_dataset(test_meta, "test", load_cached_data)

    # Datasets
    # Train: Augment = True
    train_ds = GestureDataset(*train_data, augment=True)
    # Val/Test: Augment = False
    val_ds = GestureDataset(*val_data, augment=False)
    test_ds = GestureDataset(*test_data, augment=False)

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
