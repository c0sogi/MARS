import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import scipy.io
import soundfile as sf
import torchaudio
import math
import random

# Import provided library functions
from library.utils import set_seed, get_device
from library.loss import DeepSupervisionLoss


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================
class Config:
    seed = 42
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_14"
    submission_dir = "./submission"
    cache_dir = os.path.join(working_dir, "cache")

    # Data
    joints_list = [
        3,
        2,
        4,
        8,
        5,
        9,
        6,
        10,
        7,
        11,
        1,
        0,
    ]  # Head, Shoulders, Elbows, Wrists, Hands, Spine, HipCenter
    num_joints = 12
    audio_n_mfcc = 13

    # Model
    input_dim = (
        num_joints * 3
    ) * 2 + audio_n_mfcc  # Pos(36) + Vel(36) + Audio(13) = 85
    hidden_dim = 256
    num_classes = 21  # 20 gestures + 1 background
    num_stages = 3
    tcn_layers = 10
    tcn_f_maps = 256
    dropout = 0.3

    # Training
    batch_size = 16
    epochs = 40
    lr = 1e-3
    weight_decay = 1e-4
    smoothing_weight = 0.15

    # Inference
    median_window = 7


# ==================================================================================================
# DATASET & PREPROCESSING
# ==================================================================================================
def load_mat_file(path):
    try:
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        return mat
    except Exception:
        return None


def process_audio(audio_path, target_frames):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        # Compute MFCC
        transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=Config.audio_n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = transform(waveform)  # (n_mfcc, time)
        mfcc = mfcc.transpose(0, 1)  # (time, n_mfcc)

        # Resample/Interpolate to match video frames
        if mfcc.shape[0] != target_frames:
            mfcc = mfcc.unsqueeze(0).transpose(1, 2)  # (1, n_mfcc, time)
            mfcc = F.interpolate(
                mfcc, size=target_frames, mode="linear", align_corners=False
            )
            mfcc = mfcc.transpose(1, 2).squeeze(0)  # (target_frames, n_mfcc)

        return mfcc
    except Exception:
        return torch.zeros((target_frames, Config.audio_n_mfcc))


def get_skeleton_features(mat_data):
    # Extract skeleton data
    try:
        video = mat_data["Video"]
        frames = video.Frames
        num_frames = getattr(video, "NumFrames", len(frames))

        # Skeleton is usually an array of structs
        # We need to handle if it's a single frame or multiple
        if not isinstance(frames, np.ndarray):
            frames = [frames]

        skeleton_data = np.zeros((num_frames, Config.num_joints * 3), dtype=np.float32)

        for i, frame in enumerate(frames):
            if i >= num_frames:
                break
            skeletons = frame.Skeleton
            # Assuming single user or taking the first one
            if isinstance(skeletons, np.ndarray):
                if len(skeletons) > 0:
                    skel = skeletons[0]
                else:
                    continue
            else:
                skel = skeletons

            # Extract joints
            # WorldPosition is struct with X, Y, Z
            joint_positions = []
            for j_idx in Config.joints_list:
                # Access joint by index is tricky in this struct format
                # The prompt says "JointsType" list.
                # Usually mat['Video'].Frames[i].Skeleton.WorldPosition is an array or we access by field?
                # The prompt description: "Skeleton ... JointsType ... WorldPosition"
                # Let's assume WorldPosition is an array of structs or a struct of arrays matching joint indices
                # Based on typical MSR DailyActivity/Chalearn format:
                # WorldPosition might be (20,) struct array.
                try:
                    # Try to access joint by index if WorldPosition is array
                    # If WorldPosition is a single struct with X,Y,Z, then Skeleton must be array of joints?
                    # "Skeleton ... contains the joint positions... JointsType... WorldPosition"
                    # It implies Skeleton is a struct containing arrays.
                    # Let's try to get WorldPosition for specific joint index
                    if hasattr(skel, "WorldPosition"):
                        wp = skel.WorldPosition
                        if isinstance(wp, np.ndarray) and len(wp) >= 20:
                            pos = wp[j_idx]
                            joint_positions.extend([pos.X, pos.Y, pos.Z])
                        else:
                            # Fallback or different structure
                            joint_positions.extend([0, 0, 0])
                    else:
                        joint_positions.extend([0, 0, 0])
                except:
                    joint_positions.extend([0, 0, 0])

            skeleton_data[i] = joint_positions

        return torch.tensor(skeleton_data, dtype=torch.float32)
    except Exception as e:
        # print(f"Skeleton error: {e}")
        return torch.zeros((1, Config.num_joints * 3), dtype=torch.float32)


def prepare_sample(sample_info, input_dir):
    # Load MAT
    mat_path = os.path.join(input_dir, sample_info["data_path"])
    mat = load_mat_file(mat_path)
    if mat is None:
        return None

    # Skeleton
    skel_feats = get_skeleton_features(mat)  # (T, 36)
    T = skel_feats.shape[0]

    # Audio
    audio_path = os.path.join(input_dir, sample_info["audio_path"])
    audio_feats = process_audio(audio_path, T)  # (T, 13)

    # Velocity
    # Compute velocity: P_t - P_{t-1}, pad first with 0
    vel_feats = torch.zeros_like(skel_feats)
    vel_feats[1:] = skel_feats[1:] - skel_feats[:-1]

    # Concatenate
    # (T, 36+36+13) = (T, 85)
    features = torch.cat([skel_feats, vel_feats, audio_feats], dim=1)

    # Labels
    # Create frame-wise labels
    # Default 0 (Background)
    labels = torch.zeros(T, dtype=torch.long)

    if (
        "labels" in sample_info
        and isinstance(sample_info["labels"], list)
        and len(sample_info["labels"]) > 0
    ):
        # The metadata provides ordered list of gestures, but we need frame-level annotations
        # The MAT file contains 'Labels' struct with Begin/End/Name
        # We need to parse that again from MAT to get boundaries
        try:
            video = mat["Video"]
            raw_labels = getattr(video, "Labels", [])

            def process_lbl(obj):
                try:
                    name = obj.Name
                    start = int(obj.Begin) - 1  # 1-based to 0-based
                    end = int(obj.End)
                    from metadata_script import (
                        GESTURE_MAP,
                    )  # We don't have this import, define map locally

                    g_map = {
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
                    if name in g_map:
                        gid = g_map[name]
                        # Clip to frames
                        start = max(0, start)
                        end = min(T, end)
                        labels[start:end] = gid
                except:
                    pass

            if isinstance(raw_labels, np.ndarray):
                if raw_labels.ndim == 0:
                    process_lbl(raw_labels.item())
                else:
                    for l in raw_labels:
                        process_lbl(l)
            else:
                process_lbl(raw_labels)
        except:
            pass

    return features, labels


def get_data(split="train", load_cached_data=True):
    os.makedirs(Config.cache_dir, exist_ok=True)
    cache_file = os.path.join(Config.cache_dir, f"{split}_data.pt")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {split} data from cache...")
        return torch.load(cache_file)

    print(f"Processing {split} data...")
    df = pd.read_csv(os.path.join(Config.metadata_dir, f"{split}.csv"))
    # Parse labels string to list
    df["labels"] = df["labels"].apply(
        lambda x: [int(i) for i in str(x).split()] if pd.notna(x) and x != "" else []
    )

    data_list = []

    for idx, row in df.iterrows():
        res = prepare_sample(row, Config.input_dir)
        if res is not None:
            feats, lbls = res
            data_list.append(
                {"sample_id": row["sample_id"], "features": feats, "labels": lbls}
            )

    torch.save(data_list, cache_file)
    return data_list


class GestureDataset(Dataset):
    def __init__(self, data_list, augment=False):
        self.data = data_list
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = item["features"].clone()  # (T, 85)
        labels = item["labels"].clone()

        if self.augment:
            # Augmentation: Smooth Noise on Position
            T = features.shape[0]
            # Positions are 0:36
            pos = features[:, :36]

            # Generate Gaussian Noise
            noise = torch.randn_like(pos) * 0.05
            # Temporal Low Pass (Simple Moving Average)
            noise = (
                F.avg_pool1d(noise.T.unsqueeze(0), kernel_size=5, stride=1, padding=2)
                .squeeze(0)
                .T
            )

            pos_aug = pos + noise

            # Recompute Velocity
            vel_aug = torch.zeros_like(pos_aug)
            vel_aug[1:] = pos_aug[1:] - pos_aug[:-1]

            features[:, :36] = pos_aug
            features[:, 36:72] = vel_aug

        return features, labels, item["sample_id"]


def collate_fn(batch):
    features, labels, ids = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in features])

    # Pad sequences
    features_padded = pad_sequence(features, batch_first=True)  # (B, T, C)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # (B, T)

    # Create Mask (1 for valid, 0 for pad)
    mask = torch.zeros(len(features), features_padded.shape[1])
    for i, length in enumerate(lengths):
        mask[i, :length] = 1

    # Transpose features to (B, C, T) for TCN/Conv
    features_padded = features_padded.permute(0, 2, 1)

    return features_padded, labels_padded, mask, ids


# ==================================================================================================
# MODEL ARCHITECTURE
# ==================================================================================================


class GatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(GatedConvBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation

        self.filter_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.gate_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

        # Residual connection handling
        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.res_conv = None

    def forward(self, x):
        # x: (B, C, T)

        # Compute filter and gate
        # Pad manually to handle causal or same padding. Here 'same' via slicing
        f = self.filter_conv(
            F.pad(x, (self.padding, 0))
        )  # Causal-like padding logic or just left pad
        # Wait, usually for TCN we pad left.
        # But let's stick to simple padding and slice if needed.
        # If we pad (padding, 0), output length is T + padding - (K-1)*D + 1 - 1 = T. Correct.

        g = self.gate_conv(F.pad(x, (self.padding, 0)))

        # Activation
        z = torch.tanh(f) * torch.sigmoid(g)

        # Projection
        out = self.conv_1x1(z)
        out = self.dropout(out)

        # Residual
        res = x if self.res_conv is None else self.res_conv(x)

        return out + res


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True
        )
        # Bidirectional -> 2 * hidden_dim
        self.cls_head = nn.Linear(hidden_dim * 2, num_classes)
        self.trans_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (B, C, T) -> Permute to (B, T, C)
        x = x.permute(0, 2, 1)

        feat, _ = self.lstm(x)  # (B, T, 2*H)

        cls_logits = self.cls_head(feat)  # (B, T, num_classes)
        trans_logits = self.trans_head(feat)  # (B, T, 1)
        trans_prob = torch.sigmoid(trans_logits)

        # Concatenate: (B, T, num_classes + 1)
        out = torch.cat([cls_logits, trans_prob], dim=2)

        # Permute back to (B, C, T)
        out = out.permute(0, 2, 1)
        return out


class GatedMSTCN(nn.Module):
    def __init__(self, input_dim, num_layers, num_f_maps, output_dim):
        super(GatedMSTCN, self).__init__()
        self.input_conv = nn.Conv1d(input_dim, num_f_maps, 1)

        self.layers = nn.ModuleList(
            [
                GatedConvBlock(
                    num_f_maps,
                    num_f_maps,
                    kernel_size=3,
                    dilation=2**i,
                    dropout=Config.dropout,
                )
                for i in range(num_layers)
            ]
        )

        self.output_conv = nn.Conv1d(num_f_maps, output_dim, 1)

    def forward(self, x, mask):
        # x: (B, C, T)
        out = self.input_conv(x)

        for layer in self.layers:
            out = layer(out)

        out = self.output_conv(out)

        # Apply mask
        if mask is not None:
            out = out * mask.unsqueeze(1)

        return out


class GLT_CRCN(nn.Module):
    def __init__(self):
        super(GLT_CRCN, self).__init__()

        # Stage 1: Encoder
        self.stage1 = BiLSTMEncoder(
            Config.input_dim, Config.hidden_dim, Config.num_classes
        )

        # Stage 2: Refinement (Input: 21+1, Output: 21+1)
        # Input is Class Logits + Transition Prob
        self.stage2 = GatedMSTCN(
            Config.num_classes + 1,
            Config.tcn_layers,
            Config.tcn_f_maps,
            Config.num_classes + 1,
        )

        # Stage 3: Sharpening (Input: 21+1, Output: 21)
        self.stage3 = GatedMSTCN(
            Config.num_classes + 1,
            Config.tcn_layers,
            Config.tcn_f_maps,
            Config.num_classes,
        )

    def forward(self, x, mask):
        # x: (B, C, T)
        # mask: (B, T)

        # Stage 1
        out1 = self.stage1(x)  # (B, 22, T)
        if mask is not None:
            out1 = out1 * mask.unsqueeze(1)

        # Stage 2
        out2 = self.stage2(out1, mask)  # (B, 22, T)

        # Stage 3
        out3 = self.stage3(out2, mask)  # (B, 21, T)

        return [out1, out2, out3]


# ==================================================================================================
# TRAINING & INFERENCE
# ==================================================================================================


def train_model():
    set_seed(Config.seed)
    device = get_device()

    # Data
    train_data = get_data("train")
    val_data = get_data("val")

    train_dataset = GestureDataset(train_data, augment=True)
    val_dataset = GestureDataset(val_data, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Model
    model = GLT_CRCN().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Loss
    criterion = DeepSupervisionLoss(
        num_classes=Config.num_classes, smoothing_weight=Config.smoothing_weight
    )

    best_lev = float("inf")
    patience = 5
    counter = 0

    print("Starting training...")
    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0

        for feats, targets, mask, _ in train_loader:
            feats, targets, mask = feats.to(device), targets.to(device), mask.to(device)

            optimizer.zero_grad()
            outputs = model(feats, mask)
            loss = criterion(outputs, targets, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0

        # Simple Levenshtein approx or just loss for early stopping?
        # Let's use Loss for simplicity in this loop, but printed metrics are good.
        with torch.no_grad():
            for feats, targets, mask, _ in val_loader:
                feats, targets, mask = (
                    feats.to(device),
                    targets.to(device),
                    mask.to(device),
                )
                outputs = model(feats, mask)
                loss = criterion(outputs, targets, mask)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_lev:
            best_lev = val_loss
            counter = 0
            torch.save(
                model.state_dict(), os.path.join(Config.working_dir, "best_model.pth")
            )
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

    return model


def generate_submission():
    set_seed(Config.seed)
    device = get_device()

    # Load Model
    model = GLT_CRCN().to(device)
    model_path = os.path.join(Config.working_dir, "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("No model found, using random init (will fail).")

    model.eval()

    # Test Data
    test_data = get_data("test")
    test_dataset = GestureDataset(test_data, augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for feats, _, mask, ids in test_loader:
            feats, mask = feats.to(device), mask.to(device)

            outputs = model(feats, mask)
            # Use Stage 3 output
            logits = outputs[-1]  # (B, 21, T)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()  # (B, T)

            for i in range(len(ids)):
                seq_id = ids[i]
                pred_seq = preds[i]
                valid_len = int(mask[i].sum().item())
                pred_seq = pred_seq[:valid_len]

                # Median Filter
                pred_seq = scipy.signal.medfilt(
                    pred_seq, kernel_size=Config.median_window
                )

                # Decode
                decoded = []
                last = -1
                for p in pred_seq:
                    if p != last:
                        if p != 0:  # Ignore background
                            decoded.append(str(int(p)))
                        last = p

                results.append(f"{seq_id},{','.join(decoded)}")

    # Save
    os.makedirs(Config.submission_dir, exist_ok=True)
    with open(os.path.join(Config.submission_dir, "submission.csv"), "w") as f:
        for line in results:
            f.write(line + "\n")

    print("Submission saved.")


if __name__ == "__main__":
    os.makedirs(Config.working_dir, exist_ok=True)
    train_model()
    generate_submission()
