import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    SUBMISSION_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    GESTURE_MAP,
    ID_TO_GESTURE,
    NUM_CLASSES,
    SKELETON_JOINTS,
    NUM_JOINTS,
    JOINT_DIM,
    SKELETON_SCALE_FACTOR,
    AUDIO_SAMPLE_RATE,
    NUM_MFCC,
    NUM_STAGES,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    DILATIONS,
    SEED,
    BATCH_SIZE,
    NUM_EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MEDIAN_FILTER_KERNEL,
)
from library.utils import set_seed, compute_error_rate
from library.loss import CombinedLoss

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================


class SimplifiedGatedBlock(nn.Module):
    """
    Streamlined Gated Block without 1x1 output projection.
    Z = tanh(Wf * X) * sigmoid(Wg * X)
    Y = X + Z
    """

    def __init__(self, channels, kernel_size, dilation):
        super(SimplifiedGatedBlock, self).__init__()
        self.conv = nn.Conv1d(
            channels, 2 * channels, kernel_size, padding=0, dilation=dilation
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.padding = (kernel_size - 1) * dilation // 2

    def forward(self, x):
        # x: (B, C, T)
        residual = x
        x_padded = F.pad(x, (self.padding, self.padding))
        out = self.conv(x_padded)
        f, g = torch.chunk(out, 2, dim=1)
        z = torch.tanh(f) * torch.sigmoid(g)
        z = self.dropout(z)
        return residual + z


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Bi-Directional LSTM Encoder with Class and Boundary heads.
    """

    def __init__(self, input_dim, hidden_dim):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True
        )
        self.cls_head = nn.Linear(2 * hidden_dim, NUM_CLASSES)
        self.bnd_head = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x, mask):
        # x: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        cls_logits = self.cls_head(out)
        bnd_logits = self.bnd_head(out)
        # Concat: (B, T, NUM_CLASSES + 1)
        logits = torch.cat([cls_logits, bnd_logits], dim=2)
        # Permute back: (B, C, T)
        return logits.permute(0, 2, 1)


class GatedRefinementStage(nn.Module):
    """
    Stage 2 & 3: Refinement using stacked Simplified Gated Blocks.
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GatedRefinementStage, self).__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        self.layers = nn.ModuleList(
            [
                SimplifiedGatedBlock(hidden_dim, 3, DILATIONS[i])
                for i in range(num_layers)
            ]
        )
        self.cls_head = nn.Conv1d(hidden_dim, NUM_CLASSES, 1)
        self.bnd_head = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, x, mask):
        # x: (B, input_dim, T)
        feat = self.input_proj(x)
        mask_expanded = mask.unsqueeze(1)

        for layer in self.layers:
            feat = layer(feat)
            feat = feat * mask_expanded

        cls_logits = self.cls_head(feat)
        bnd_logits = self.bnd_head(feat)
        return torch.cat([cls_logits, bnd_logits], dim=1)


class SSG_CRCN(nn.Module):
    """
    Streamlined Supervised Gated-Cascaded Recurrent-Convolutional Network.
    """

    def __init__(self):
        super(SSG_CRCN, self).__init__()
        # Input: (12 joints * 3 coords) * 2 (pos+vel) + 13 MFCC
        self.input_dim = (NUM_JOINTS * JOINT_DIM) * 2 + NUM_MFCC

        self.stage1 = BiLSTMEncoder(self.input_dim, HIDDEN_DIM)

        refine_in_dim = NUM_CLASSES + 1
        self.stage2 = GatedRefinementStage(refine_in_dim, HIDDEN_DIM, NUM_LAYERS)
        self.stage3 = GatedRefinementStage(refine_in_dim, HIDDEN_DIM, NUM_LAYERS)

    def forward(self, x, mask):
        outputs = []

        # Stage 1
        s1_logits = self.stage1(x, mask)
        outputs.append(s1_logits)

        # Prepare S2 input
        s1_cls = F.softmax(s1_logits[:, :NUM_CLASSES, :], dim=1)
        s1_bnd = torch.sigmoid(s1_logits[:, NUM_CLASSES:, :])
        s1_probs = torch.cat([s1_cls, s1_bnd], dim=1) * mask.unsqueeze(1)

        # Stage 2
        s2_logits = self.stage2(s1_probs, mask)
        outputs.append(s2_logits)

        # Prepare S3 input
        s2_cls = F.softmax(s2_logits[:, :NUM_CLASSES, :], dim=1)
        s2_bnd = torch.sigmoid(s2_logits[:, NUM_CLASSES:, :])
        s2_probs = torch.cat([s2_cls, s2_bnd], dim=1) * mask.unsqueeze(1)

        # Stage 3
        s3_logits = self.stage3(s2_probs, mask)
        outputs.append(s3_logits)

        return outputs


# =============================================================================
# DATA PROCESSING
# =============================================================================


def process_sample(sample_info):
    """
    Loads and processes a single sample: Skeleton, Audio, Labels.
    """
    # 1. Load Skeleton Data
    mat_path = os.path.join(INPUT_DIR, sample_info["data_path"])
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        video = mat["Video"]
        num_frames = getattr(video, "NumFrames", 0)
        frames = getattr(video, "Frames", [])

        # Extract WorldPosition for selected joints
        # Frames is an array of Skeleton structures
        # We need to handle if Frames is a single object or array
        if not isinstance(frames, np.ndarray) and not isinstance(frames, list):
            frames = [frames]

        skeleton_data = np.zeros((num_frames, NUM_JOINTS, JOINT_DIM), dtype=np.float32)

        for t, frame in enumerate(frames):
            if t >= num_frames:
                break
            skel = getattr(frame, "Skeleton", None)
            if skel is None:
                continue

            # skel is struct with WorldPosition
            # WorldPosition is struct with X, Y, Z
            # We need to iterate over joints
            # The structure of Skeleton in the mat file:
            # It contains an array of joints? Or fields for each joint?
            # Description says: Skeleton structure contains JointsType, WorldPosition...
            # Usually in these datasets, Skeleton is an array of joints.
            # Let's assume standard Kinect structure where Skeleton is array of joints.

            # Actually, based on description: "Skeleton Frame: An array of Skeleton structures is contained within a Skeletons array... The format of a Skeleton structure... JointsType... WorldPosition"
            # This implies 'Frames' contains 'Skeleton' which contains joints.
            # Let's try to access WorldPosition directly if it's an array of joints.

            # Robust extraction:
            try:
                # Assuming skel is an array of joints
                for j_idx, joint_idx in enumerate(SKELETON_JOINTS):
                    # Access joint by index
                    if isinstance(skel, np.ndarray) or isinstance(skel, list):
                        joint = skel[joint_idx]
                    else:
                        # If it's not iterable, maybe it's a single struct? Unlikely for multiple joints.
                        # Fallback for safety
                        continue

                    wp = getattr(joint, "WorldPosition", None)
                    if wp:
                        skeleton_data[t, j_idx, 0] = wp.X
                        skeleton_data[t, j_idx, 1] = wp.Y
                        skeleton_data[t, j_idx, 2] = wp.Z
            except Exception:
                pass

    except Exception as e:
        # Fallback empty
        return None

    # Preprocessing Skeleton
    # 1. Center around HipCenter (Index 0)
    hip_center = skeleton_data[:, 0:1, :]  # (T, 1, 3)
    skeleton_data = skeleton_data - hip_center

    # 2. Scale
    skeleton_data = skeleton_data * SKELETON_SCALE_FACTOR

    # 3. Compute Velocity
    velocity = np.zeros_like(skeleton_data)
    velocity[1:] = skeleton_data[1:] - skeleton_data[:-1]

    # Flatten: (T, J*3)
    skel_flat = skeleton_data.reshape(num_frames, -1)
    vel_flat = velocity.reshape(num_frames, -1)

    # 2. Load Audio
    audio_path = os.path.join(INPUT_DIR, sample_info["audio_path"])
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        # Resample if needed (though config says 16k, data might vary)
        if sample_rate != AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, AUDIO_SAMPLE_RATE)
            waveform = resampler(waveform)

        # Compute MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=AUDIO_SAMPLE_RATE,
            n_mfcc=NUM_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)
        mfcc = mfcc.mean(dim=0)  # Average over channels if stereo -> (n_mfcc, time)

        # Align to Video Frames
        # MFCC shape: (Features, AudioFrames)
        # Target: (Features, VideoFrames)
        mfcc = mfcc.unsqueeze(0)  # (1, F, T_audio)
        mfcc_aligned = F.interpolate(
            mfcc, size=num_frames, mode="linear", align_corners=False
        )
        mfcc_aligned = mfcc_aligned.squeeze(0).permute(1, 0).numpy()  # (T_video, F)

    except Exception:
        # Fallback zero audio
        mfcc_aligned = np.zeros((num_frames, NUM_MFCC), dtype=np.float32)

    # Concatenate Features
    features = np.concatenate([skel_flat, vel_flat, mfcc_aligned], axis=1)  # (T, D)

    # 3. Process Labels (Frame-wise)
    # Convert sequence of gesture IDs to frame-wise labels
    # Labels provided as list of IDs. We need start/end frames from .mat
    # The metadata csv has 'labels' column but that's just the sequence.
    # We need frame-level annotations for training.
    # The .mat file has 'Labels' struct with Begin/End/Name.

    frame_labels = np.zeros(num_frames, dtype=np.int64)  # Default 0 (Background)
    boundaries = np.zeros(num_frames, dtype=np.float32)

    if "labels" in sample_info and sample_info["labels"]:
        # Re-parse mat for label details
        try:
            video = mat["Video"]
            labels_raw = getattr(video, "Labels", [])

            def process_lbl(obj):
                try:
                    name = obj.Name
                    if name in GESTURE_MAP:
                        gid = GESTURE_MAP[name]
                        start = int(obj.Begin) - 1  # 1-based to 0-based
                        end = int(obj.End)
                        # Clip
                        start = max(0, start)
                        end = min(num_frames, end)

                        frame_labels[start:end] = gid
                        # Boundary at start
                        if start < num_frames:
                            boundaries[start] = 1.0
                except:
                    pass

            if isinstance(labels_raw, np.ndarray):
                if labels_raw.ndim == 0:
                    process_lbl(labels_raw.item())
                else:
                    for l in labels_raw:
                        process_lbl(l)
            else:
                process_lbl(labels_raw)
        except:
            pass

    return features.astype(np.float32), frame_labels, boundaries


def prepare_dataset(metadata_path, cache_name, load_cached=True):
    cache_path = os.path.join(WORKING_DIR, f"{cache_name}.npz")

    if load_cached and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["features"], data["labels"], data["boundaries"], data["ids"]

    print(f"Processing data from {metadata_path}")
    df = pd.read_csv(metadata_path)
    # Parse labels string to list
    df["labels_seq"] = df["labels"].apply(
        lambda x: [int(i) for i in str(x).split()] if pd.notna(x) and x != "" else []
    )

    features_list = []
    labels_list = []
    boundaries_list = []
    ids_list = []

    for _, row in df.iterrows():
        res = process_sample(row)
        if res is None:
            continue
        feat, lbl, bnd = res
        features_list.append(feat)
        labels_list.append(lbl)
        boundaries_list.append(bnd)
        ids_list.append(row["sample_id"])

    # Save object array because lengths differ
    features_arr = np.array(features_list, dtype=object)
    labels_arr = np.array(labels_list, dtype=object)
    boundaries_arr = np.array(boundaries_list, dtype=object)
    ids_arr = np.array(ids_list, dtype=object)

    np.savez_compressed(
        cache_path,
        features=features_arr,
        labels=labels_arr,
        boundaries=boundaries_arr,
        ids=ids_arr,
    )
    return features_arr, labels_arr, boundaries_arr, ids_arr


class GestureDataset(Dataset):
    def __init__(self, features, labels, boundaries):
        self.features = features
        self.labels = labels
        self.boundaries = boundaries

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.labels[idx]),
            torch.from_numpy(self.boundaries[idx]),
        )


def collate_fn(batch):
    features, labels, boundaries = zip(*batch)
    lengths = [f.shape[0] for f in features]
    max_len = max(lengths)

    # Pad
    feat_pad = torch.zeros(len(features), features[0].shape[1], max_len)
    lbl_pad = torch.zeros(len(labels), max_len, dtype=torch.long)
    bnd_pad = torch.zeros(len(boundaries), max_len, dtype=torch.float)
    mask = torch.zeros(len(features), max_len, dtype=torch.bool)

    for i, (f, l, b) in enumerate(zip(features, labels, boundaries)):
        end = lengths[i]
        feat_pad[i, :, :end] = f.permute(1, 0)  # (T, D) -> (D, T)
        lbl_pad[i, :end] = l
        bnd_pad[i, :end] = b
        mask[i, :end] = 1

    return feat_pad, lbl_pad, bnd_pad, mask


# =============================================================================
# TRAINING & INFERENCE
# =============================================================================


def train_model():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_feat, train_lbl, train_bnd, _ = prepare_dataset(
        TRAIN_METADATA_PATH, "train_data"
    )
    val_feat, val_lbl, val_bnd, _ = prepare_dataset(VAL_METADATA_PATH, "val_data")

    train_loader = DataLoader(
        GestureDataset(train_feat, train_lbl, train_bnd),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        GestureDataset(val_feat, val_lbl, val_bnd),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    model = SSG_CRCN().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = CombinedLoss().to(device)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for feats, lbls, bnds, mask in train_loader:
            feats, lbls, bnds, mask = (
                feats.to(device),
                lbls.to(device),
                bnds.to(device),
                mask.to(device),
            )

            optimizer.zero_grad()
            outputs = model(feats, mask)
            loss, _ = criterion(outputs, lbls, bnds, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for feats, lbls, bnds, mask in val_loader:
                feats, lbls, bnds, mask = (
                    feats.to(device),
                    lbls.to(device),
                    bnds.to(device),
                    mask.to(device),
                )
                outputs = model(feats, mask)
                loss, _ = criterion(outputs, lbls, bnds, mask)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(WORKING_DIR, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    return model


def generate_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data
    test_feat, test_lbl, test_bnd, test_ids = prepare_dataset(
        TEST_METADATA_PATH, "test_data"
    )

    # Load Model
    model = SSG_CRCN().to(device)
    model.load_state_dict(
        torch.load(os.path.join(WORKING_DIR, "best_model.pth"), map_location=device)
    )
    model.eval()

    predictions = []

    # Process one by one (batch size 1 for simplicity in inference)
    with torch.no_grad():
        for i in range(len(test_ids)):
            feat = torch.from_numpy(test_feat[i]).unsqueeze(0)  # (1, T, D)
            # Permute for model: (1, D, T)
            feat = feat.permute(0, 2, 1).to(device)
            mask = torch.ones(1, feat.shape[2], dtype=torch.bool).to(device)

            outputs = model(feat, mask)
            final_stage_logits = outputs[-1]  # (1, C+1, T)
            cls_logits = final_stage_logits[:, :NUM_CLASSES, :]
            probs = F.softmax(cls_logits, dim=1)
            preds = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()  # (T,)

            # Post-processing
            # 1. Median Filter
            if MEDIAN_FILTER_KERNEL > 1:
                from scipy.signal import medfilt

                preds = medfilt(preds, kernel_size=MEDIAN_FILTER_KERNEL)

            # 2. Decode to sequence
            # Collapse repeats and remove background (0)
            decoded = []
            prev = -1
            for p in preds:
                if p != prev:
                    if p != 0:
                        decoded.append(int(p))
                    prev = p

            predictions.append(decoded)

    # Save Submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    with open(submission_path, "w") as f:
        for sid, pred in zip(test_ids, predictions):
            pred_str = ",".join(map(str, pred))
            f.write(f"{sid},{pred_str}\n")
    print(f"Submission saved to {submission_path}")


def main():
    train_model()
    generate_submission()


# Execute
main()
