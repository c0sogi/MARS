import os
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config


class SETIDataset(Dataset):
    """
    Custom Dataset for SETI Technosignature Detection.
    Handles loading, splitting, synchronized augmentation, and padding of spectrograms.
    """

    def __init__(self, df, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, target, file_path).
            mode (str): 'train', 'val', or 'test'. Controls augmentation behavior.
        """
        self.df = df.copy()
        self.mode = mode

        # Handle Debugging Mode
        if Config.DEBUG:
            print(
                f"DEBUG MODE: Subsetting dataset to {Config.DEBUG_SAMPLE_SIZE} samples."
            )
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

        self.file_paths = self.df["file_path"].values

        # Handle targets: Test set might have placeholder targets
        if "target" in self.df.columns:
            self.targets = self.df["target"].values
        else:
            self.targets = np.zeros(len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Construct Path and Load Data
        # file_paths are relative, e.g., "train/0/xxxx.npy"
        full_path = os.path.join(Config.INPUT_ROOT, self.file_paths[idx])

        try:
            # Load spectrogram: Shape (6, 273, 256)
            # 6 positions (ABACAD), 273 freq bins, 256 time steps
            spectrogram = np.load(full_path).astype(np.float32)
        except FileNotFoundError:
            # Fallback for robustness (though paths should be verified)
            print(f"Warning: File not found {full_path}. Returning zeros.")
            spectrogram = np.zeros((6, 273, 256), dtype=np.float32)

        # 2. Split into On-Target and Off-Target Streams
        # On-Target (A): Indices 0, 2, 4
        # Off-Target (B, C, D): Indices 1, 3, 5
        on_source = spectrogram[[0, 2, 4], :, :]  # Shape (3, 273, 256)
        off_source = spectrogram[[1, 3, 5], :, :]  # Shape (3, 273, 256)

        # 3. Synchronized Augmentation (Train only)
        if self.mode == "train":
            # Random Horizontal Flip (Time axis = 2)
            if np.random.rand() < 0.5:
                on_source = np.flip(on_source, axis=2)
                off_source = np.flip(off_source, axis=2)

            # Random Vertical Flip (Frequency axis = 1)
            if np.random.rand() < 0.5:
                on_source = np.flip(on_source, axis=1)
                off_source = np.flip(off_source, axis=1)

        # 4. Padding (Strictly AFTER Augmentation)
        # Target shape: Config.INPUT_SIZE = (288, 256)
        # Current shape: (3, 273, 256)
        # We pad the frequency dimension (axis 1) from 273 to 288.
        # We pad at the end of the axis to keep artifacts consistent.

        target_h, target_w = Config.INPUT_SIZE
        current_h, current_w = on_source.shape[1], on_source.shape[2]

        pad_h = max(0, target_h - current_h)
        pad_w = max(0, target_w - current_w)

        # Pad format: ((before_c, after_c), (before_h, after_h), (before_w, after_w))
        # We only pad the end of height and width (if needed)
        padding = ((0, 0), (0, pad_h), (0, pad_w))

        if pad_h > 0 or pad_w > 0:
            # mode='constant', constant_values=0 is default for np.pad but explicit is better
            on_source = np.pad(on_source, padding, mode="constant", constant_values=0)
            off_source = np.pad(off_source, padding, mode="constant", constant_values=0)

            # Crop if larger (safety check, though unlikely given dataset specs)
            on_source = on_source[:, :target_h, :target_w]
            off_source = off_source[:, :target_h, :target_w]

        # 5. Convert to Tensors
        # Copy ensures negative strides from flipping are handled
        tensor_on = torch.from_numpy(on_source.copy()).float()
        tensor_off = torch.from_numpy(off_source.copy()).float()
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return tensor_on, tensor_off, target
