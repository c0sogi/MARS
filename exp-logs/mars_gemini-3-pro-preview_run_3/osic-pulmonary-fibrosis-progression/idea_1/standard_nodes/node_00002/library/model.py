import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import cv2

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.loss import LaplaceLogLikelihoodLoss

# Attempt to import pydicom for image processing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    print("Warning: pydicom not found. Image features will be zeroed out.")


# ====================================================
# Dataset
# ====================================================
class OSICDataset(Dataset):
    def __init__(self, df, mode="train", cache_dir=Config.CACHE_DIR):
        self.mode = mode
        self.cache_dir = cache_dir

        # Feature Engineering: Extract Baseline Information
        # We need to ensure every row has access to the patient's baseline characteristics.
        # For training data, we group by Patient.

        if mode == "test":
            # For test, df is the test.csv (baselines).
            # We need to generate rows for every Patient_Week in sample_submission.
            sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

            # Parse Patient and Weeks from Patient_Week ID
            # ID format: ID00419637202311204720264_12
            sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
            sub_df["Weeks"] = sub_df["Patient_Week"].apply(
                lambda x: int(x.split("_")[1])
            )

            # Merge baseline info from test.csv
            # test.csv has [Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus] (Baseline values)
            # We rename baseline columns to avoid collision with the target 'Weeks'
            base_df = df.rename(
                columns={
                    "Weeks": "Baseline_Weeks",
                    "FVC": "Baseline_FVC",
                    "Percent": "Baseline_Percent",
                    "Age": "Baseline_Age",
                }
            )

            self.data = sub_df.merge(base_df, on="Patient", how="left")

        else:
            # For train/val, we have the full history.
            # We need to identify the baseline (first visit) for each patient.
            # We assume the provided metadata/train.csv is already split, but we need baseline info.

            # We need the original train.csv to find global baselines for these patients
            # or we can assume the earliest visit in the provided df is the baseline.
            # To be robust, we'll calculate baseline from the provided df per patient.

            # Sort by Weeks to find the first visit
            df_sorted = df.sort_values(["Patient", "Weeks"])
            baseline_df = df_sorted.groupby("Patient").first().reset_index()
            baseline_df = baseline_df[
                ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
            ]
            baseline_df = baseline_df.rename(
                columns={
                    "Weeks": "Baseline_Weeks",
                    "FVC": "Baseline_FVC",
                    "Percent": "Baseline_Percent",
                    "Age": "Baseline_Age",
                    "Sex": "Baseline_Sex",
                    "SmokingStatus": "Baseline_SmokingStatus",
                }
            )

            # Merge back to the original df
            self.data = df.merge(baseline_df, on="Patient", how="left")

            # For training, 'Sex' and 'SmokingStatus' in the row are the same as baseline (static features)
            # We just ensure we have the columns.
            self.data["Sex"] = self.data["Baseline_Sex"]
            self.data["SmokingStatus"] = self.data["Baseline_SmokingStatus"]

        # Normalization Constants (Approximate from EDA)
        self.norm_stats = {
            "Age": (67.0, 15.0),
            "Percent": (77.0, 20.0),
            "FVC": (2650.0, 800.0),
            "Weeks": (30.0, 25.0),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_id = row["Patient"]

        # 1. Tabular Features
        # Numerical: Relative Weeks, Baseline FVC, Baseline Percent, Age
        weeks_rel = row["Weeks"] - row["Baseline_Weeks"]

        # Normalize
        feat_weeks = weeks_rel / 100.0  # Simple scaling for relative time
        feat_age = (row["Baseline_Age"] - self.norm_stats["Age"][0]) / self.norm_stats[
            "Age"
        ][1]
        feat_fvc = (row["Baseline_FVC"] - self.norm_stats["FVC"][0]) / self.norm_stats[
            "FVC"
        ][1]
        feat_pct = (
            row["Baseline_Percent"] - self.norm_stats["Percent"][0]
        ) / self.norm_stats["Percent"][1]

        # Categorical: Sex (Binary), Smoking (One-Hot)
        # Sex: Male=0, Female=1
        feat_sex = 1.0 if row["Sex"] == "Female" else 0.0

        # Smoking: Ex-smoker, Never smoked, Currently smokes
        # [Is_Ex, Is_Never, Is_Current]
        feat_smoke = [0.0, 0.0, 0.0]
        if row["SmokingStatus"] == "Ex-smoker":
            feat_smoke[0] = 1.0
        elif row["SmokingStatus"] == "Never smoked":
            feat_smoke[1] = 1.0
        elif row["SmokingStatus"] == "Currently smokes":
            feat_smoke[2] = 1.0

        tabular = np.array(
            [feat_weeks, feat_fvc, feat_pct, feat_age, feat_sex] + feat_smoke,
            dtype=np.float32,
        )

        # 2. Image Features (Representative Slice)
        img_tensor = self._load_image(patient_id)

        # 3. Target
        if self.mode != "test":
            target_fvc = float(row["FVC"])
            return img_tensor, tabular, torch.tensor([target_fvc], dtype=torch.float32)
        else:
            return (
                img_tensor,
                tabular,
                torch.tensor([0.0], dtype=torch.float32),
            )  # Dummy target

    def _load_image(self, patient_id):
        # Check cache
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")

        if os.path.exists(cache_path):
            try:
                img_array = np.load(cache_path)
                # Convert to tensor (C, H, W)
                img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(
                    2, 0, 1
                )
                return img_tensor
            except Exception:
                pass  # Fallback to processing if load fails

        # Process from scratch
        if HAS_PYDICOM:
            # Determine directory
            if self.mode == "test":
                dcm_dir = os.path.join(Config.TEST_DIR, patient_id)
            else:
                dcm_dir = os.path.join(Config.TRAIN_DIR, patient_id)

            if not os.path.exists(dcm_dir):
                # Fallback for missing directories
                img_array = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32
                )
            else:
                files = [f for f in os.listdir(dcm_dir) if f.endswith(".dcm")]
                if not files:
                    img_array = np.zeros(
                        (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32
                    )
                else:
                    # Heuristic: Find slice with max lung area
                    best_slice = None
                    max_lung_pixels = -1

                    # Limit scan to middle 50% to save time if many files, or check all if feasible
                    # Checking all for better accuracy as per "Representative Slice" idea
                    files.sort()  # Ensure some order

                    for f in files:
                        try:
                            path = os.path.join(dcm_dir, f)
                            dcm = pydicom.dcmread(path)
                            img = dcm.pixel_array.astype(np.float32)

                            # Rescale to HU
                            intercept = getattr(dcm, "RescaleIntercept", 0)
                            slope = getattr(dcm, "RescaleSlope", 1)
                            img = img * slope + intercept

                            # Count lung pixels
                            lung_pixels = np.sum(
                                (img >= Config.LUNG_MIN_HU)
                                & (img <= Config.LUNG_MAX_HU)
                            )

                            if lung_pixels > max_lung_pixels:
                                max_lung_pixels = lung_pixels
                                best_slice = img
                        except Exception:
                            continue

                    if best_slice is None:
                        img_array = np.zeros(
                            (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32
                        )
                    else:
                        # Windowing
                        img = best_slice
                        img_min = Config.WINDOW_CENTER - Config.WINDOW_WIDTH // 2
                        img_max = Config.WINDOW_CENTER + Config.WINDOW_WIDTH // 2
                        img[img < img_min] = img_min
                        img[img > img_max] = img_max

                        # Normalize 0-1
                        img = (img - img_min) / (img_max - img_min)

                        # Resize
                        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

                        # Stack to 3 channels
                        img_array = np.stack([img, img, img], axis=-1)
        else:
            # No pydicom, return zeros
            img_array = np.zeros(
                (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32
            )

        # Save to cache
        try:
            np.save(cache_path, img_array)
        except Exception:
            pass

        img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1)
        return img_tensor


# ====================================================
# Model
# ====================================================
class OSICModel(nn.Module):
    def __init__(self, tabular_input_dim=8):
        super(OSICModel, self).__init__()

        # 1. Image Branch (CNN)
        # Load EfficientNet-B0, remove classification head (num_classes=0)
        self.cnn = timm.create_model(
            Config.MODEL_NAME, pretrained=Config.PRETRAINED, num_classes=0
        )

        # Get feature dimension (usually 1280 for B0)
        self.img_feature_dim = self.cnn.num_features

        if Config.FREEZE_BACKBONE:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # 2. Tabular Branch (MLP)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )
        self.tab_feature_dim = 64

        # 3. Fusion & Head
        fusion_dim = self.img_feature_dim + self.tab_feature_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, Config.OUTPUT_DIM),  # Output: [FVC, Sigma]
        )

    def forward(self, img, tabular):
        # Image Features
        # img: (B, 3, H, W)
        img_emb = self.cnn(img)  # (B, 1280)

        # Tabular Features
        tab_emb = self.tabular_mlp(tabular)  # (B, 64)

        # Concatenate
        combined = torch.cat([img_emb, tab_emb], dim=1)

        # Prediction
        output = self.head(combined)

        return output


# ====================================================
# Training & Execution
# ====================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = 0
    count = 0

    for imgs, tabs, targets in loader:
        imgs = imgs.to(device)
        tabs = tabs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(imgs, tabs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        loss_meter += loss.item() * imgs.size(0)
        count += imgs.size(0)

    return loss_meter / count


def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = 0
    metric_meter = 0
    count = 0

    with torch.no_grad():
        for imgs, tabs, targets in loader:
            imgs = imgs.to(device)
            tabs = tabs.to(device)
            targets = targets.to(device)

            preds = model(imgs, tabs)
            loss = criterion(preds, targets)

            # Calculate Metric
            fvc_pred = preds[:, 0].cpu().numpy()
            sigma_pred = preds[:, 1].cpu().numpy()
            y_true = targets.cpu().numpy().flatten()

            # Ensure sigma is positive for metric calc (model outputs raw logits/values)
            sigma_pred = np.abs(sigma_pred)

            score = laplace_log_likelihood_metric(y_true, fvc_pred, sigma_pred)

            loss_meter += loss.item() * imgs.size(0)
            metric_meter += score * imgs.size(0)
            count += imgs.size(0)

    return loss_meter / count, metric_meter / count


def run_experiment():
    seed_everything(Config.SEED)
    Config.setup()

    print(f"Starting Experiment: {Config.EXPERIMENT_NAME}")

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        train_df = train_df.head(50)
        val_df = val_df.head(20)

    # Datasets
    train_dataset = OSICDataset(train_df, mode="train")
    val_dataset = OSICDataset(val_df, mode="val")
    test_dataset = OSICDataset(test_df, mode="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Model Setup
    device = torch.device(Config.DEVICE)
    model = OSICModel().to(device)

    criterion = LaplaceLogLikelihoodLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Training Loop
    best_loss = float("inf")
    patience_counter = 0

    print("Training Started...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metric = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val Metric: {val_metric:.5f}"
        )

        # Checkpoint & Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(Config.MODEL_CHECKPOINT_DIR, "best_model.pth"),
            )
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # Inference on Test Set
    print("Generating Submission...")
    model.load_state_dict(
        torch.load(
            os.path.join(Config.MODEL_CHECKPOINT_DIR, "best_model.pth"),
            map_location=device,
        )
    )
    model.eval()

    all_preds = []
    all_sigmas = []

    with torch.no_grad():
        for imgs, tabs, _ in test_loader:
            imgs = imgs.to(device)
            tabs = tabs.to(device)

            preds = model(imgs, tabs)

            fvc = preds[:, 0].cpu().numpy()
            sigma = preds[:, 1].cpu().numpy()

            all_preds.extend(fvc)
            all_sigmas.extend(sigma)

    # Prepare Submission DataFrame
    # Note: The test_dataset.data is ordered same as the loader because shuffle=False
    sub_df = test_dataset.data.copy()
    sub_df["FVC"] = np.array(all_preds)
    sub_df["Confidence"] = np.abs(np.array(all_sigmas))  # Ensure positive

    # Apply Post-Processing (Clipping per metric definition)
    # Note: The metric clips sigma at 70, but the submission format just asks for confidence.
    # It is beneficial to clip confidence at 70 for the file as well to match the evaluation logic.
    sub_df["Confidence"] = np.maximum(sub_df["Confidence"], Config.SIGMA_CLIP)

    # Save
    final_sub = sub_df[["Patient_Week", "FVC", "Confidence"]]
    final_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
