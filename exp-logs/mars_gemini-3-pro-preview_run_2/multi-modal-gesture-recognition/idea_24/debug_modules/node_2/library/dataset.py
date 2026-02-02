import os
import torch
import numpy as np
import pandas as pd
import scipy.io
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from library.config import INPUT_DIR, GESTURE_MAP, BOUNDARY_SIGMA, NUM_CLASSES, SEED
from library.features import compute_all_features, process_data
from library.utils import load_metadata, set_seed


class GestureDataset(Dataset):
    def __init__(self, split="train", augment=False, load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            augment (bool): Whether to apply geometric augmentation.
            load_cached_data (bool): Whether to use cached pre-processed data.
        """
        set_seed(SEED)
        self.split = split
        self.augment = augment

        # 1. Load Processed Data (Skeletons, Audio, IDs)
        # process_data handles caching internally
        data_dict = process_data(split, load_cached_data=load_cached_data)

        self.sample_ids = data_dict["sample_ids"]
        self.skeletons = data_dict["skeletons"]
        self.audio = data_dict["audio"]

        # 2. Load Metadata for File Path Mapping
        # We need this to access the .mat files for frame-level label extraction
        meta_df = load_metadata(split)
        # Create a map from sample_id to data_path
        self.id_to_datapath = pd.Series(
            meta_df.data_path.values, index=meta_df.sample_id
        ).to_dict()

    def __len__(self):
        return len(self.sample_ids)

    def _get_frame_labels(self, data_path, num_frames):
        """
        Parses the .mat file to generate frame-level class and boundary targets.

        Args:
            data_path (str): Relative path to the .mat file.
            num_frames (int): Total number of frames in the sequence (T).

        Returns:
            class_target (torch.Tensor): (T,) Long tensor with class IDs.
            boundary_target (torch.Tensor): (T,) Float tensor with soft boundaries.
        """
        class_target = torch.zeros(num_frames, dtype=torch.long)
        boundary_target = torch.zeros(num_frames, dtype=torch.float32)

        # If test set, we don't have labels (or they are empty), return zeros
        if self.split == "test":
            return class_target, boundary_target

        full_path = os.path.join(INPUT_DIR, data_path)

        try:
            mat = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)
            if "Video" not in mat:
                return class_target, boundary_target

            video = mat["Video"]
            labels_raw = getattr(video, "Labels", [])

            # Helper to process a single label object
            def process_label_obj(obj):
                try:
                    name = obj.Name
                    if name not in GESTURE_MAP:
                        return

                    g_id = GESTURE_MAP[name]
                    # MATLAB is 1-based, Python is 0-based
                    start_frame = int(obj.Begin) - 1
                    end_frame = int(obj.End) - 1

                    # Clip to valid range
                    start_frame = max(0, min(start_frame, num_frames - 1))
                    end_frame = max(0, min(end_frame, num_frames - 1))

                    if start_frame > end_frame:
                        return

                    # 1. Fill Class Target (Hard Labels)
                    # We assume the gesture is active from start to end (inclusive)
                    class_target[start_frame : end_frame + 1] = g_id

                    # 2. Fill Boundary Target (Soft Gaussian)
                    # We place a Gaussian bump at the start and end of the gesture
                    # These represent "transitions"
                    for t_center in [start_frame, end_frame]:
                        # Define a window for the Gaussian (e.g., +/- 3 sigma)
                        window_radius = int(3 * BOUNDARY_SIGMA)
                        t_min = max(0, t_center - window_radius)
                        t_max = min(num_frames, t_center + window_radius + 1)

                        indices = torch.arange(t_min, t_max, dtype=torch.float32)
                        # Gaussian function: exp(- (x - mu)^2 / (2 * sigma^2))
                        gaussian = torch.exp(
                            -((indices - t_center) ** 2) / (2 * BOUNDARY_SIGMA**2)
                        )

                        # Accumulate (using max to handle overlapping transitions cleanly)
                        boundary_target[t_min:t_max] = torch.maximum(
                            boundary_target[t_min:t_max], gaussian
                        )

                except AttributeError:
                    pass

            # Handle different shapes of labels_raw (scalar, array, empty)
            if isinstance(labels_raw, np.ndarray):
                if labels_raw.ndim == 0:
                    process_label_obj(labels_raw.item())
                else:
                    for l in labels_raw:
                        process_label_obj(l)
            else:
                process_label_obj(labels_raw)

        except Exception as e:
            # In case of read error, return zeros (background)
            pass

        return class_target, boundary_target

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # 1. Get Features
        # skeletons: (T, 12, 3), audio: (T, 13)
        skel = self.skeletons[idx]
        aud = self.audio[idx]

        # Compute fused features (T, InputDim)
        # Handles augmentation if self.augment is True
        features = compute_all_features(skel, aud, augment=self.augment)

        num_frames = features.shape[0]

        # 2. Get Targets
        data_path = self.id_to_datapath.get(sample_id, "")
        class_target, boundary_target = self._get_frame_labels(data_path, num_frames)

        # 3. Create Mask (all ones for valid data)
        mask = torch.ones(num_frames, dtype=torch.bool)

        return {
            "features": features,  # (T, D)
            "class_target": class_target,  # (T,)
            "boundary_target": boundary_target,  # (T,)
            "mask": mask,  # (T,)
            "sample_id": sample_id,
        }


def collate_fn(batch):
    """
    Collates a list of dataset items into a batch.
    Pads sequences to the maximum length in the batch.
    """
    # Extract lists
    features_list = [item["features"] for item in batch]
    class_target_list = [item["class_target"] for item in batch]
    boundary_target_list = [item["boundary_target"] for item in batch]
    mask_list = [item["mask"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]

    # Pad sequences (Batch First)
    # Features pad with 0
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)

    # Class targets pad with 0 (Background)
    class_target_padded = pad_sequence(
        class_target_list, batch_first=True, padding_value=0
    )

    # Boundary targets pad with 0 (No boundary)
    boundary_target_padded = pad_sequence(
        boundary_target_list, batch_first=True, padding_value=0.0
    )

    # Mask pads with False
    mask_padded = pad_sequence(mask_list, batch_first=True, padding_value=False)

    return {
        "features": features_padded,
        "class_target": class_target_padded,
        "boundary_target": boundary_target_padded,
        "mask": mask_padded,
        "sample_ids": sample_ids,
    }
