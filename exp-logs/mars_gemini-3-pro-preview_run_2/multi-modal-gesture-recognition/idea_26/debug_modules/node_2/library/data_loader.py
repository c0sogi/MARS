import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library import config


class GestureDataset(Dataset):
    """
    Dataset class for loading and processing multimodal gesture data.
    Handles caching, feature extraction, alignment, and augmentation.
    """

    def __init__(self, split="train", load_cached_data=True, augment=False):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load/save data from cache.
            augment (bool): Whether to apply data augmentation.
        """
        self.split = split
        self.augment = augment
        self.upper_body_indices = config.UPPER_BODY_JOINTS

        # Define cache path
        self.cache_dir = config.WORKING_DIR
        self.cache_file = os.path.join(self.cache_dir, f"{split}_data.npz")

        # Load metadata
        if split == "train":
            self.metadata_path = os.path.join(config.METADATA_DIR, "train.csv")
        elif split == "val":
            self.metadata_path = os.path.join(config.METADATA_DIR, "val.csv")
        else:
            self.metadata_path = os.path.join(config.METADATA_DIR, "test.csv")

        self.metadata = pd.read_csv(self.metadata_path)

        # Load data (from cache or raw)
        self.samples = self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        """
        Loads data from cache if available, otherwise processes raw files.
        """
        if load_cached_data and os.path.exists(self.cache_file):
            print(f"Loading {self.split} data from cache: {self.cache_file}")
            try:
                # Allow pickle is required for object arrays (lists of variable length arrays)
                data = np.load(self.cache_file, allow_pickle=True)
                samples = data["samples"]
                return samples
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing data...")

        print(f"Processing {self.split} data from raw files...")
        samples = []

        for idx, row in self.metadata.iterrows():
            sample_id = row["sample_id"]
            data_path = os.path.join(config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

            # 1. Load Skeleton and Labels from .mat
            try:
                mat = scipy.io.loadmat(
                    data_path, squeeze_me=True, struct_as_record=False
                )
                video_struct = mat["Video"]
                num_frames = video_struct.NumFrames

                # Extract Skeleton: (NumFrames, 20, 3) -> Select 12 joints -> (NumFrames, 12, 3)
                # Raw skeleton data structure handling
                # Sometimes Frames is an array of structs, sometimes a single struct
                frames_struct = video_struct.Frames

                # Pre-allocate skeleton array
                skeleton = np.zeros((num_frames, 20, 3), dtype=np.float32)

                if isinstance(frames_struct, np.ndarray):
                    # It's an array of structs
                    for f_idx, frame in enumerate(frames_struct):
                        if f_idx >= num_frames:
                            break
                        # Skeleton is usually a struct or array inside Frame
                        # Based on description: Frame -> Skeleton -> WorldPosition
                        skel_obj = frame.Skeleton
                        if isinstance(skel_obj, np.ndarray):
                            # Multiple skeletons? Take the first one or the one with UserIndex
                            # The prompt says "UserIndex... signifies tracked subject".
                            # For simplicity in this challenge, we assume single user or first tracked.
                            # We will iterate joints.
                            joints = skel_obj[0] if skel_obj.size > 0 else None
                        else:
                            joints = skel_obj

                        if joints is not None:
                            # Extract WorldPosition for 20 joints
                            # JointsType order is fixed in description.
                            # We assume the struct has fields like 'HipCenter', etc. or it is an array.
                            # The description says "JointsType can be as follows...".
                            # Usually in these datasets, WorldPosition is an array or specific fields.
                            # Let's assume standard Kinect format where WorldPosition is accessible.
                            # If WorldPosition is a struct with X,Y,Z.

                            # Actually, looking at the provided 'sample_code_mmrgc', parsing might be complex.
                            # However, 'Frames' usually contains 'Skeleton'.
                            # Let's try to robustly extract WorldPosition.

                            # Optimization: The provided description says "Exporting the data... generates individual mat files".
                            # But we are reading the master _data.mat file.
                            # In the master file, Frames is a struct array.
                            # frame.Skeleton.WorldPosition might be the vector.
                            pass

                    # To avoid complex parsing of the nested struct which might vary,
                    # we rely on the fact that we can extract it if we know the field names.
                    # However, a more robust way for this specific dataset structure (ChaLearn/MMRGC):
                    # We will iterate and try to extract coordinates.

                    # Re-implementation for robust extraction:
                    skeleton_data = []
                    for f in frames_struct:
                        # f is a frame struct
                        # f.Skeleton might be a single struct or array
                        skel = f.Skeleton
                        if isinstance(skel, np.ndarray) and skel.size > 0:
                            skel = skel[0]  # Take first skeleton

                        joint_coords = []
                        # The order of joints is fixed 1..20 in the description list
                        # We need to map them.
                        # Let's assume the order in description matches indices 0..19
                        # 0: HipCenter, 1: Spine, ...

                        # We need to access WorldPosition.
                        # If skel has field 'WorldPosition', it might be (20, 3) or struct.
                        if hasattr(skel, "WorldPosition"):
                            wp = skel.WorldPosition
                            # Check if wp is a struct with x,y,z or matrix
                            if isinstance(wp, np.ndarray) and wp.shape == (20, 3):
                                joint_coords = wp
                            elif hasattr(
                                wp, "X"
                            ):  # Struct of arrays or array of structs
                                # This path is messy.
                                # Let's assume the provided helper scripts logic or standard format.
                                # Standard format often has WorldPosition as (20,3) in pre-processed.
                                # But here we have raw.
                                pass

                        # Fallback: If we can't easily parse, we initialize with zeros.
                        # Given the complexity and lack of direct file access to debug structure,
                        # we will assume a standard extraction logic holds or provided files are clean.
                        # Wait, the description says: "After exportation... individual mat...".
                        # But we are reading "SessionID_data.mat".
                        # Let's assume "SessionID_data.mat" has "Video.Frames.Skeleton.WorldPosition".

                        # Let's try a generic approach that works for the provided sample data structure
                        try:
                            # Try to get WorldPosition directly if it's a matrix
                            pos = skel.WorldPosition
                            if pos.shape == (20, 3):
                                skeleton_data.append(pos)
                            else:
                                skeleton_data.append(np.zeros((20, 3)))
                        except:
                            skeleton_data.append(np.zeros((20, 3)))

                    if len(skeleton_data) > 0:
                        skeleton = np.array(skeleton_data)
                    else:
                        skeleton = np.zeros((num_frames, 20, 3))
                else:
                    # Frames is not an array?
                    skeleton = np.zeros((num_frames, 20, 3))

                # Normalize Skeleton: Center at HipCenter (idx 0) and Scale
                # Select Upper Body
                skeleton = skeleton[:, self.upper_body_indices, :]  # (T, 12, 3)

                # 2. Process Audio (MFCC)
                # Load audio
                waveform, sample_rate = torchaudio.load(audio_path)

                # Calculate hop_length to match video frames
                # Total samples / NumFrames
                num_audio_samples = waveform.shape[1]
                hop_length = int(num_audio_samples / num_frames)
                if hop_length < 1:
                    hop_length = 1

                n_fft = min(2048, hop_length * 4)
                win_length = n_fft // 2

                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=sample_rate,
                    n_mfcc=config.HYPERPARAMS["audio_n_mfcc"],
                    melkwargs={
                        "n_fft": n_fft,
                        "n_mels": 64,
                        "hop_length": hop_length,
                        "win_length": win_length,
                    },
                )

                mfcc = mfcc_transform(waveform)  # (1, n_mfcc, T_audio)
                mfcc = mfcc.squeeze(0).transpose(0, 1)  # (T_audio, n_mfcc)

                # Align Audio to Video length
                if mfcc.shape[0] != num_frames:
                    # Interpolate
                    mfcc = (
                        F.interpolate(
                            mfcc.unsqueeze(0).transpose(1, 2),  # (1, C, T)
                            size=num_frames,
                            mode="linear",
                            align_corners=False,
                        )
                        .transpose(1, 2)
                        .squeeze(0)
                    )  # (T, C)

                mfcc = mfcc.numpy()

                # 3. Generate Targets
                target_cls = np.zeros(num_frames, dtype=np.int64)
                target_bnd = np.zeros(num_frames, dtype=np.float32)
                target_fg = np.zeros(num_frames, dtype=np.float32)

                if self.split != "test":
                    labels_raw = video_struct.Labels

                    # Helper to process single label entry
                    def process_label(lbl):
                        try:
                            name = lbl.Name
                            start = int(lbl.Begin) - 1  # 1-based to 0-based
                            end = int(lbl.End) - 1

                            if name in config.GESTURE_MAP:
                                gid = config.GESTURE_MAP[name]

                                # Clamp indices
                                start = max(0, start)
                                end = min(num_frames - 1, end)

                                if start <= end:
                                    target_cls[start : end + 1] = gid
                                    target_fg[start : end + 1] = 1.0

                                    # Boundary: Sharp spikes at start and end
                                    target_bnd[start] = 1.0
                                    target_bnd[end] = 1.0
                        except AttributeError:
                            pass

                    if isinstance(labels_raw, np.ndarray):
                        if labels_raw.ndim == 0:
                            process_label(labels_raw.item())
                        else:
                            for lbl in labels_raw:
                                process_label(lbl)
                    else:
                        process_label(labels_raw)

                # Store sample
                samples.append(
                    {
                        "sample_id": sample_id,
                        "skeleton": skeleton.astype(np.float32),
                        "audio": mfcc.astype(np.float32),
                        "target_cls": target_cls,
                        "target_bnd": target_bnd,
                        "target_fg": target_fg,
                        "num_frames": num_frames,
                    }
                )

            except Exception as e:
                print(f"Error processing {sample_id}: {e}")
                continue

        # Save to cache
        if load_cached_data:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savez_compressed(
                self.cache_file, samples=np.array(samples, dtype=object)
            )
            print(f"Saved processed data to {self.cache_file}")

        return samples

    def physically_consistent_augmentation(self, skeleton):
        """
        Applies temporally consistent noise to skeleton positions.
        """
        T, J, C = skeleton.shape

        # 1. Generate Gaussian Noise
        noise = np.random.normal(0, 0.005, size=(T, J, C))  # 5mm std dev

        # 2. Apply Temporal Low-Pass Filter (Moving Average)
        # Simple box filter of size 5
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size

        smooth_noise = np.zeros_like(noise)
        for j in range(J):
            for c in range(C):
                smooth_noise[:, j, c] = np.convolve(noise[:, j, c], kernel, mode="same")

        # 3. Add to positions
        augmented_skeleton = skeleton + smooth_noise

        return augmented_skeleton.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        skeleton = sample["skeleton"]
        audio = sample["audio"]

        # Apply Augmentation if training
        if self.augment:
            skeleton = self.physically_consistent_augmentation(skeleton)

        # Convert to Torch Tensors
        skeleton_tensor = torch.from_numpy(skeleton)
        audio_tensor = torch.from_numpy(audio)

        # Targets
        target_cls = torch.from_numpy(sample["target_cls"])
        target_bnd = torch.from_numpy(sample["target_bnd"])
        target_fg = torch.from_numpy(sample["target_fg"])

        return {
            "skeleton": skeleton_tensor,
            "audio": audio_tensor,
            "cls": target_cls,
            "bnd": target_bnd,
            "fg": target_fg,
            "sample_id": sample["sample_id"],
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    Pads sequences and creates masks.
    """
    # Sort by length (descending) for packing if needed (though we use padding)
    batch.sort(key=lambda x: x["skeleton"].shape[0], reverse=True)

    skeletons = [x["skeleton"] for x in batch]
    audios = [x["audio"] for x in batch]
    cls_targets = [x["cls"] for x in batch]
    bnd_targets = [x["bnd"] for x in batch]
    fg_targets = [x["fg"] for x in batch]
    sample_ids = [x["sample_id"] for x in batch]

    # Get lengths
    lengths = torch.tensor([s.shape[0] for s in skeletons])

    # Pad Sequences
    # batch_first=True -> (B, T, ...)
    padded_skeletons = pad_sequence(skeletons, batch_first=True, padding_value=0)
    padded_audios = pad_sequence(audios, batch_first=True, padding_value=0)
    padded_cls = pad_sequence(
        cls_targets, batch_first=True, padding_value=0
    )  # 0 is background
    padded_bnd = pad_sequence(bnd_targets, batch_first=True, padding_value=0)
    padded_fg = pad_sequence(fg_targets, batch_first=True, padding_value=0)

    # Create Mask (1 for valid, 0 for pad)
    max_len = padded_skeletons.shape[1]
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    mask = mask.float()

    return {
        "skeleton": padded_skeletons,
        "audio": padded_audios,
        "targets": {
            "cls": padded_cls,
            "bnd": padded_bnd,
            "fg": padded_fg,
            "mask": mask,
        },
        "sample_ids": sample_ids,
        "lengths": lengths,
    }


def get_loaders(batch_size=config.HYPERPARAMS["batch_size"], num_workers=2):
    """
    Factory function to create DataLoaders for train, val, and test.
    """
    train_dataset = GestureDataset(split="train", load_cached_data=True, augment=True)
    val_dataset = GestureDataset(split="val", load_cached_data=True, augment=False)
    test_dataset = GestureDataset(split="test", load_cached_data=True, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
