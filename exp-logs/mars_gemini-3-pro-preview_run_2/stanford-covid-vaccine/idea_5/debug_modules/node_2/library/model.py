import os
import ast
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# 1. Helper Modules (Layers)
# =========================================================================


class SeparableConv1d(nn.Module):
    """
    Depthwise Separable Convolution.
    Consists of a depthwise convolution (spatial) followed by a pointwise convolution (channel mixing).
    Reduces parameters compared to standard Conv1d.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding="same",
        bias=True,
    ):
        super(SeparableConv1d, self).__init__()

        # Depthwise: groups = in_channels
        # Note: padding='same' requires stride=1 in PyTorch.
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )

        # Pointwise: kernel_size=1
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    Injects absolute position information into the sequence.
    """

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, C, L) for easy concatenation/addition with (B, C, L)
        self.pe = pe.transpose(0, 1).unsqueeze(0)
        self.register_buffer("pe_buffer", self.pe)

    def forward(self, x):
        # x shape: (Batch, Channels, Seq_Len)
        # Returns positional encoding sliced to seq_len
        seq_len = x.size(2)
        return self.pe_buffer[:, :, :seq_len].repeat(x.size(0), 1, 1)


class ResidualBlock(nn.Module):
    """
    Dilated Residual Block using Separable Convolutions.
    Structure: SepConv -> ReLU -> Dropout -> SepConv -> ReLU -> Dropout
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(ResidualBlock, self).__init__()

        self.conv1 = SeparableConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding="same"
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = SeparableConv1d(
            out_channels, out_channels, kernel_size, dilation=dilation, padding="same"
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Skip connection adjustment if dimensions differ
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out  # Activation usually happens inside the block or next block, here we keep it linear after sum or add relu?
        # Standard ResNet adds ReLU after sum. TCN usually keeps it as is. We'll leave it linear to allow gradient flow.


# =========================================================================
# 2. Main Model Architecture
# =========================================================================


class HybridNet(nn.Module):
    """
    Positional Partner-Aware Separable Hybrid Network.

    Features:
    - Spatially augmented input (Sequence + Structure + Loop + Partners + Distance + Positional).
    - Depthwise Separable Dilated TCN Backbone.
    - Multi-scale fusion.
    - BiGRU global aggregation.
    """

    def __init__(self):
        super(HybridNet, self).__init__()

        # 1. Determine Input Dimension
        # Basic One-Hot: Seq(4) + Struct(3) + Loop(7) = 14
        self.in_channels = 14

        if Config.USE_PARTNER_FEATURES:
            self.in_channels += 4  # Partner base identity

        if Config.USE_DISTANCE_FEATURES:
            self.in_channels += 1  # Normalized distance

        if Config.USE_POSITIONAL_ENCODING:
            self.pe_dim = 16
            self.pos_encoder = PositionalEncoding(self.pe_dim, max_len=Config.SEQ_LEN)
            self.in_channels += self.pe_dim

        # 2. Initial Convolution
        self.stem = nn.Conv1d(self.in_channels, Config.HIDDEN_DIM, kernel_size=1)

        # 3. Dilated TCN Backbone (Separable)
        self.blocks = nn.ModuleList()
        for i in range(Config.NUM_LAYERS):
            dilation = 2**i
            self.blocks.append(
                ResidualBlock(
                    Config.HIDDEN_DIM,
                    Config.HIDDEN_DIM,
                    Config.KERNEL_SIZE,
                    dilation,
                    Config.DROPOUT,
                )
            )

        # 4. BiGRU
        # Input to GRU is concatenation of all block outputs (Multi-Scale Fusion)
        # Fusion dim = NUM_LAYERS * HIDDEN_DIM
        self.fusion_dim = Config.NUM_LAYERS * Config.HIDDEN_DIM
        self.gru = nn.GRU(
            input_size=self.fusion_dim,
            hidden_size=Config.HIDDEN_DIM
            // 2,  # Bidirectional -> Output dim = HIDDEN_DIM
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 5. Output Head
        self.head = nn.Linear(Config.HIDDEN_DIM, 5)  # 5 Targets

    def forward(self, x):
        # x shape: (Batch, Basic_Channels, Seq_Len)
        # If Positional Encoding is enabled, it's generated internally and concatenated

        if Config.USE_POSITIONAL_ENCODING:
            pe = self.pos_encoder(x)  # (Batch, 16, Seq_Len)
            x = torch.cat([x, pe], dim=1)

        # Stem
        x = self.stem(x)  # (B, Hidden, L)

        # Backbone with Multi-Scale Fusion
        block_outputs = []
        current_out = x

        for block in self.blocks:
            current_out = block(current_out)
            block_outputs.append(current_out)

        # Concatenate all scales: (B, Layers*Hidden, L)
        fused = torch.cat(block_outputs, dim=1)

        # Prepare for GRU: (B, L, C)
        fused = fused.permute(0, 2, 1)

        # BiGRU
        gru_out, _ = self.gru(fused)  # (B, L, Hidden)

        # Head
        logits = self.head(gru_out)  # (B, L, 5)

        return logits


# =========================================================================
# 3. Data Processing & Dataset
# =========================================================================


class RNADataset(Dataset):
    def __init__(self, inputs, targets=None, ids=None):
        self.inputs = inputs
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Inputs: (Channels, Seq_Len)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        sample = {"inputs": x}

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["targets"] = y

        if self.ids is not None:
            sample["ids"] = self.ids[idx]

        return sample


def get_structure_pairs(structure):
    """
    Parses dot-bracket structure to find pairs.
    Returns a dictionary mapping index -> partner_index.
    """
    pairs = {}
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i
    return pairs


def process_sequence_features(df):
    """
    Generates the spatially augmented input tensor for the dataframe.
    Returns: numpy array of shape (N, Channels, Seq_Len)
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Define mappings
    base_map = {b: i for i, b in enumerate(Config.BASES)}
    struct_map = {s: i for i, s in enumerate(Config.STRUCTURES)}
    loop_map = {l: i for i, l in enumerate(Config.LOOP_TYPES)}

    # Calculate channel counts
    c_seq = len(Config.BASES)
    c_struct = len(Config.STRUCTURES)
    c_loop = len(Config.LOOP_TYPES)
    c_partner = len(Config.BASES) if Config.USE_PARTNER_FEATURES else 0
    c_dist = 1 if Config.USE_DISTANCE_FEATURES else 0
    # Positional encoding is handled inside the model forward pass, not here

    total_channels = c_seq + c_struct + c_loop + c_partner + c_dist

    # Pre-allocate
    features = np.zeros((num_samples, total_channels, seq_len), dtype=np.float32)

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # 1. Basic One-Hot Features
        for i, char in enumerate(seq):
            if char in base_map:
                features[idx, base_map[char], i] = 1.0

        for i, char in enumerate(struct):
            if char in struct_map:
                features[idx, c_seq + struct_map[char], i] = 1.0

        for i, char in enumerate(loop):
            if char in loop_map:
                features[idx, c_seq + c_struct + loop_map[char], i] = 1.0

        # 2. Structural Features (Partner & Distance)
        if Config.USE_PARTNER_FEATURES or Config.USE_DISTANCE_FEATURES:
            pairs = get_structure_pairs(struct)

            offset = c_seq + c_struct + c_loop

            for i in range(seq_len):
                if i in pairs:
                    partner_idx = pairs[i]

                    # Partner Identity
                    if Config.USE_PARTNER_FEATURES:
                        partner_base = seq[partner_idx]
                        if partner_base in base_map:
                            features[idx, offset + base_map[partner_base], i] = 1.0

                    # Distance
                    if Config.USE_DISTANCE_FEATURES:
                        dist_offset = offset + (
                            c_partner if Config.USE_PARTNER_FEATURES else 0
                        )
                        # Normalize distance by sequence length
                        dist_val = abs(i - partner_idx) / seq_len
                        features[idx, dist_offset, i] = dist_val

    return features


def load_data(mode="train", load_cached_data=True):
    """
    Loads data, processes features, and returns a Dataset object.
    Handles caching to ./working/idea_5/data_cache/

    Args:
        mode: 'train', 'val', or 'test'
        load_cached_data: If True, attempts to load from .npy files
    """
    cache_dir = os.path.join(Config.WORKING_DIR, "data_cache")
    os.makedirs(cache_dir, exist_ok=True)

    inputs_path = os.path.join(cache_dir, f"{mode}_inputs.npy")
    targets_path = os.path.join(cache_dir, f"{mode}_targets.npy")
    ids_path = os.path.join(cache_dir, f"{mode}_ids.npy")

    # Try loading cache
    if load_cached_data and os.path.exists(inputs_path) and os.path.exists(ids_path):
        # For test set, targets might not exist
        if mode == "test" or os.path.exists(targets_path):
            print(f"Loading cached {mode} data from {cache_dir}...")
            inputs = np.load(inputs_path)
            ids = np.load(ids_path, allow_pickle=True)
            targets = np.load(targets_path) if os.path.exists(targets_path) else None
            return RNADataset(inputs, targets, ids)

    # Process from scratch
    print(f"Processing {mode} data from metadata...")

    if mode == "train":
        csv_path = Config.TRAIN_METADATA_PATH
    elif mode == "val":
        csv_path = Config.VAL_METADATA_PATH
    else:
        csv_path = Config.TEST_METADATA_PATH

    df = pd.read_csv(csv_path)

    # Generate Inputs
    inputs = process_sequence_features(df)
    ids = df["id"].values

    # Generate Targets
    targets = None
    if mode in ["train", "val"]:
        # Targets are stored as stringified lists in CSV
        target_cols = Config.ALL_TARGET_COLS
        targets = np.zeros(
            (len(df), Config.SEQ_LEN, len(target_cols)), dtype=np.float32
        )

        for i, col in enumerate(target_cols):
            # Parse string "[0.1, 0.2...]" -> list
            # Note: The CSVs in metadata/ are generated by pandas.
            # If they contain lists, read_csv reads them as strings.
            # We use ast.literal_eval.
            # However, for speed, we can assume valid formatting.

            # Vectorized apply is faster than loop
            parsed_col = df[col].apply(
                lambda x: (
                    np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x)
                )
            )

            # Fill tensor
            for row_idx, val_array in enumerate(parsed_col):
                # val_array length is seq_scored (68), rest is 0 or ignored by mask
                length = min(len(val_array), Config.SEQ_LEN)
                targets[row_idx, :length, i] = val_array[:length]

    # Save to cache
    np.save(inputs_path, inputs)
    np.save(ids_path, ids)
    if targets is not None:
        np.save(targets_path, targets)

    return RNADataset(inputs, targets, ids)


def get_dataloaders(load_cached_data=True):
    """
    Returns train_loader, val_loader, test_loader
    """
    train_ds = load_data("train", load_cached_data)
    val_ds = load_data("val", load_cached_data)
    test_ds = load_data("test", load_cached_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
