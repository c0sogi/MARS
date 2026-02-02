import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import cv2
from library.config import Config

# Set fixed seeds for reproducibility
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


class FeatureEngineer:
    def __init__(self):
        # Mapping based on the dataset description
        self.joint_map = {
            "HipCenter": 0,
            "Spine": 1,
            "ShoulderCenter": 2,
            "Head": 3,
            "ShoulderLeft": 4,
            "ElbowLeft": 5,
            "WristLeft": 6,
            "HandLeft": 7,
            "ShoulderRight": 8,
            "ElbowRight": 9,
            "WristRight": 10,
            "HandRight": 11,
            "HipLeft": 12,
            "KneeLeft": 13,
            "AnkleLeft": 14,
            "FootLeft": 15,
            "HipRight": 16,
            "KneeRight": 17,
            "AnkleRight": 18,
            "FootRight": 19,
        }

        # Audio transformation pipeline
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,  # Assuming 16k based on analysis, will resample if needed
            n_mfcc=Config.AUDIO_N_MFCC,
        )

    def _extract_skeleton(self, mat_path):
        """
        Parses .mat file to extract raw skeleton coordinates.
        Returns: (NumFrames, 20, 3) numpy array.
        """
        try:
            # Load mat file with struct_as_record=False to allow dot notation
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

            if not hasattr(mat, "Video") or not hasattr(mat.Video, "Frames"):
                raise ValueError(f"Invalid mat structure in {mat_path}")

            frames = mat.Video.Frames

            # Handle single frame case
            if not isinstance(frames, (np.ndarray, list)):
                frames = [frames]

            num_frames = len(frames)
            # Initialize with zeros: (T, Joints, 3)
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            for t, frame in enumerate(frames):
                # Check if Skeleton data exists for this frame
                if hasattr(frame, "Skeleton") and isinstance(
                    frame.Skeleton, (np.ndarray, list)
                ):
                    joints = frame.Skeleton

                    # Handle single joint case (unlikely but possible)
                    if not isinstance(joints, (np.ndarray, list)):
                        joints = [joints]

                    for joint in joints:
                        if hasattr(joint, "JointsType") and hasattr(
                            joint, "WorldPosition"
                        ):
                            j_type = str(joint.JointsType)
                            if j_type in self.joint_map:
                                j_idx = self.joint_map[j_type]
                                pos = joint.WorldPosition
                                # WorldPosition might be struct with X,Y,Z
                                if (
                                    hasattr(pos, "X")
                                    and hasattr(pos, "Y")
                                    and hasattr(pos, "Z")
                                ):
                                    skeleton_data[t, j_idx, 0] = pos.X
                                    skeleton_data[t, j_idx, 1] = pos.Y
                                    skeleton_data[t, j_idx, 2] = pos.Z

                # Simple imputation for missing tracking: copy previous frame
                if t > 0 and np.all(skeleton_data[t] == 0):
                    skeleton_data[t] = skeleton_data[t - 1]

            return skeleton_data

        except Exception as e:
            print(f"Error processing skeleton {mat_path}: {e}")
            # Return a minimal valid array to prevent crash, though this sample is likely bad
            return np.zeros((1, Config.NUM_JOINTS, 3), dtype=np.float32)

    def _normalize_skeleton(self, skeleton):
        """
        Performs root-relative normalization.
        Subtracts HipCenter (Index 0) from all joints.
        """
        # skeleton: (T, 20, 3)
        # HipCenter is index 0
        root_pos = skeleton[:, 0:1, :]  # (T, 1, 3)
        normalized = skeleton - root_pos
        return normalized

    def _compute_derivatives(self, skeleton):
        """
        Computes velocity and acceleration.
        Returns: velocity (T, 20, 3), acceleration (T, 20, 3)
        """
        # Velocity: P_t - P_{t-1}
        # Prepend the first frame to maintain shape
        velocity = np.diff(skeleton, axis=0, prepend=skeleton[0:1])

        # Acceleration: V_t - V_{t-1}
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])

        return velocity, acceleration

    def _extract_audio(self, audio_path, target_num_frames):
        """
        Extracts MFCC features and aligns them to video frames.
        Returns: (target_num_frames, n_mfcc)
        """
        try:
            if not os.path.exists(audio_path):
                return np.zeros(
                    (target_num_frames, Config.AUDIO_N_MFCC), dtype=np.float32
                )

            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if necessary (though analysis showed 16k)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )
                waveform = resampler(waveform)

            # Compute MFCC: Output shape (Channel, n_mfcc, time) -> squeeze -> (n_mfcc, time)
            mfcc = self.mfcc_transform(waveform).squeeze(0).numpy()

            # Resize to match video frames
            # cv2.resize expects (width, height). Here width=time, height=channels
            # We want output (target_num_frames, n_mfcc)
            # So we pass dsize=(target_num_frames, n_mfcc)

            if mfcc.shape[1] > 0:
                mfcc_aligned = cv2.resize(
                    mfcc,
                    (target_num_frames, Config.AUDIO_N_MFCC),
                    interpolation=cv2.INTER_LINEAR,
                )
                # Output of resize is (height, width) -> (n_mfcc, target_num_frames) ?
                # Wait, cv2.resize(src, dsize=(W, H)). Result is (H, W).
                # Here dsize=(target_num_frames, n_mfcc). Result is (n_mfcc, target_num_frames).
                # We want (target_num_frames, n_mfcc).
                # So we transpose.
                return mfcc_aligned.T
            else:
                return np.zeros(
                    (target_num_frames, Config.AUDIO_N_MFCC), dtype=np.float32
                )

        except Exception as e:
            # print(f"Error processing audio {audio_path}: {e}")
            return np.zeros((target_num_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

    def _create_label_sequence(self, num_frames, labels_json):
        """
        Converts sparse label annotations to dense frame-wise labels.
        Default class is 0 (Background).
        """
        dense_labels = np.zeros(num_frames, dtype=np.int64)

        if not labels_json:
            return dense_labels

        try:
            labels_list = (
                json.loads(labels_json) if isinstance(labels_json, str) else labels_json
            )

            for label in labels_list:
                start = max(0, int(label["begin"]) - 1)  # 1-based to 0-based
                end = min(num_frames, int(label["end"]))
                gid = int(label["id"])

                if start < end:
                    dense_labels[start:end] = gid

        except Exception as e:
            print(f"Error parsing labels: {e}")

        return dense_labels

    def process_dataset(
        self, metadata_path, cache_path, load_cached_data=True, max_samples=None
    ):
        """
        Main pipeline to process a dataset split.
        Args:
            metadata_path: Path to CSV.
            cache_path: Path to save/load .npz.
            load_cached_data: Whether to use cache.
            max_samples: Limit number of samples for debugging.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}...")
            try:
                data = np.load(cache_path, allow_pickle=True)
                return {
                    "features": data["features"],
                    "labels": data["labels"],
                    "seq_lengths": data["seq_lengths"],
                    "sample_ids": data["sample_ids"],
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Process from Scratch
        print(f"Processing dataset from {metadata_path}...")
        df = pd.read_csv(metadata_path)

        if max_samples is not None:
            df = df.head(max_samples)

        all_features_list = []
        all_labels_list = []
        seq_lengths = []
        sample_ids = []

        for idx, row in df.iterrows():
            sample_id = row["sample_id"]
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            # --- Skeleton Processing ---
            # 1. Extract
            skeleton = self._extract_skeleton(mat_path)  # (T, 20, 3)
            num_frames = skeleton.shape[0]

            # 2. Normalize
            norm_skeleton = self._normalize_skeleton(skeleton)

            # 3. Derivatives
            features_list = [norm_skeleton.reshape(num_frames, -1)]  # (T, 60)

            if Config.USE_VELOCITY:
                vel, acc = self._compute_derivatives(norm_skeleton)
                features_list.append(vel.reshape(num_frames, -1))
                if Config.USE_ACCELERATION:
                    features_list.append(acc.reshape(num_frames, -1))

            # --- Audio Processing ---
            audio_feat = self._extract_audio(audio_path, num_frames)  # (T, 13)
            features_list.append(audio_feat)

            # --- Concatenate Features ---
            # Shape: (T, 60 + 60 + 60 + 13) = (T, 193)
            sample_features = np.concatenate(features_list, axis=1).astype(np.float32)

            # --- Label Processing ---
            sample_labels = self._create_label_sequence(num_frames, row["labels"])

            # Store
            all_features_list.append(sample_features)
            all_labels_list.append(sample_labels)
            seq_lengths.append(num_frames)
            sample_ids.append(sample_id)

        # Concatenate all into single arrays
        if all_features_list:
            final_features = np.concatenate(all_features_list, axis=0)
            final_labels = np.concatenate(all_labels_list, axis=0)
            final_seq_lengths = np.array(seq_lengths, dtype=np.int32)
            final_sample_ids = np.array(sample_ids, dtype=str)
        else:
            final_features = np.empty((0, Config.INPUT_DIM), dtype=np.float32)
            final_labels = np.empty((0,), dtype=np.int64)
            final_seq_lengths = np.empty((0,), dtype=np.int32)
            final_sample_ids = np.empty((0,), dtype=str)

        # 3. Save Cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=final_features,
            labels=final_labels,
            seq_lengths=final_seq_lengths,
            sample_ids=final_sample_ids,
        )
        print(f"Saved processed features to {cache_path}")

        return {
            "features": final_features,
            "labels": final_labels,
            "seq_lengths": final_seq_lengths,
            "sample_ids": final_sample_ids,
        }
