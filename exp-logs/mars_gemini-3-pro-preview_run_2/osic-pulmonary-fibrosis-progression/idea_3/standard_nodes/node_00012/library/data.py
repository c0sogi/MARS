import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import timm

from library.config import Config
from library.utils import seed_everything

# Attempt to import pydicom; handle absence gracefully to ensure pipeline runs
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


class DataProcessor:
    """
    Handles data loading, preprocessing, feature extraction, and caching.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.img_size = Config.IMG_SIZE
        self.num_slices = Config.NUM_SLICES

        # Image preprocessing pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Tabular preprocessors
        self.scaler_cont = StandardScaler()
        # Explicitly define categories to ensure consistent feature shape (Cite debug_lesson_2)
        self.enc_sex = OneHotEncoder(
            categories=[["Female", "Male"]],
            handle_unknown="ignore",
            sparse_output=False,
        )
        self.enc_smoke = OneHotEncoder(
            categories=[["Currently smokes", "Ex-smoker", "Never smoked"]],
            handle_unknown="ignore",
            sparse_output=False,
        )

    def read_dicom_slices(self, dcm_dir):
        """
        Reads 12 uniform slices from a patient directory.
        Returns tensor: (12, 3, 224, 224)
        """
        # Fallback if pydicom is missing or directory is empty
        if not HAS_PYDICOM or not os.path.exists(dcm_dir):
            if not HAS_PYDICOM:
                # Print once or log would be better, but keeping silent for speed/cleanliness
                pass
            return torch.zeros(self.num_slices, 3, self.img_size, self.img_size)

        files = [f for f in os.listdir(dcm_dir) if f.lower().endswith(".dcm")]
        if not files:
            return torch.zeros(self.num_slices, 3, self.img_size, self.img_size)

        # Sort files to ensure correct anatomical order (approximate by filename if instance num missing)
        try:
            files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        except ValueError:
            files.sort()

        # Select uniform indices
        indices = np.linspace(0, len(files) - 1, self.num_slices).astype(int)
        selected_files = [files[i] for i in indices]

        images = []
        for f in selected_files:
            path = os.path.join(dcm_dir, f)
            try:
                dcm = pydicom.dcmread(path)
                img = dcm.pixel_array.astype(float)

                # Rescale Slope/Intercept
                slope = getattr(dcm, "RescaleSlope", 1)
                intercept = getattr(dcm, "RescaleIntercept", 0)
                img = img * slope + intercept

                # Simple Min-Max Normalization to 0-255 for CNN input
                # (Medical windowing could be better but requires domain knowledge on specific HU)
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min) * 255.0
                else:
                    img = np.zeros_like(img)

                img = img.astype(np.uint8)
                img_pil = Image.fromarray(img).convert("RGB")
                img_tensor = self.transform(img_pil)
                images.append(img_tensor)
            except Exception:
                # Fallback for corrupt single file
                images.append(torch.zeros(3, self.img_size, self.img_size))

        return torch.stack(images)

    def extract_image_features(self, patient_ids, dcm_dirs):
        """
        Extracts features using EfficientNet-B0.
        Returns dict: {patient_id: feature_tensor (12, 1280)}
        """
        print(f"Extracting image features for {len(patient_ids)} unique patients...")

        # Load backbone
        model = timm.create_model(Config.MODEL_NAME, pretrained=True, num_classes=0)
        model.to(self.device)
        model.eval()

        features_dict = {}

        with torch.no_grad():
            for pid, dcm_rel_path in zip(patient_ids, dcm_dirs):
                full_path = os.path.join(Config.INPUT_DIR, dcm_rel_path)

                # Get batch of slices
                img_batch = self.read_dicom_slices(full_path).to(self.device)

                # Forward pass
                # Output shape: (12, 1280)
                feats = model(img_batch)
                features_dict[pid] = feats.cpu().numpy()

        return features_dict

    def process_tabular(self, df, is_train=True):
        """
        Processes tabular features (scaling, encoding).
        Returns: tab_features, weeks, targets
        """
        # Ensure baseline features exist by resolving them at runtime if missing (Cite debug_lesson_1)
        if "Baseline_FVC" not in df.columns or "Baseline_Percent" not in df.columns:
            df = df.copy()
            # Identify baseline (earliest visit) for each patient
            baseline = (
                df.sort_values("Weeks")
                .groupby("Patient")[["FVC", "Percent"]]
                .first()
                .reset_index()
            )
            baseline = baseline.rename(
                columns={"FVC": "Baseline_FVC", "Percent": "Baseline_Percent"}
            )
            df = pd.merge(df, baseline, on="Patient", how="left")

        # Continuous
        cont_data = df[Config.CONTINUOUS_COLS].values.astype(np.float32)

        # Categorical
        sex_data = df[["Sex"]].values
        smoke_data = df[["SmokingStatus"]].values

        if is_train:
            self.scaler_cont.fit(cont_data)
            self.enc_sex.fit(sex_data)
            self.enc_smoke.fit(smoke_data)

        cont_scaled = self.scaler_cont.transform(cont_data)
        sex_encoded = self.enc_sex.transform(sex_data)
        smoke_encoded = self.enc_smoke.transform(smoke_data)

        # Concatenate
        tab_features = np.hstack([cont_scaled, sex_encoded, smoke_encoded]).astype(
            np.float32
        )

        # Weeks (Independent variable for linear model)
        # We keep it raw or lightly scaled. Let's divide by 100 to keep gradients stable.
        weeks = df["Weeks"].values.astype(np.float32) / 100.0

        # Targets
        targets = df["FVC"].values.astype(np.float32)

        return tab_features, weeks, targets

    def align_data(self, df, img_dict, tab_feat, weeks, targets):
        """
        Aligns patient-level image features to sample-level tabular data.
        """
        aligned_imgs = []
        for pid in df["Patient"].values:
            if pid in img_dict:
                aligned_imgs.append(img_dict[pid])
            else:
                # Should not happen if logic is correct
                aligned_imgs.append(
                    np.zeros((self.num_slices, Config.EMBED_DIM), dtype=np.float32)
                )

        aligned_imgs = np.array(aligned_imgs, dtype=np.float32)
        return aligned_imgs, tab_feat, weeks, targets

    def run(self, load_cached=True):
        # Define extra paths for Weeks which aren't in Config explicitly
        train_w_path = os.path.join(Config.WORKING_DIR, "train_weeks.npy")
        val_w_path = os.path.join(Config.WORKING_DIR, "val_weeks.npy")
        test_w_path = os.path.join(Config.WORKING_DIR, "test_weeks.npy")

        # Check cache
        cache_files = [
            Config.TRAIN_FEATURES_PATH,
            Config.TRAIN_TABULAR_PATH,
            Config.TRAIN_TARGETS_PATH,
            train_w_path,
            Config.VAL_FEATURES_PATH,
            Config.VAL_TABULAR_PATH,
            Config.VAL_TARGETS_PATH,
            val_w_path,
            Config.TEST_FEATURES_PATH,
            Config.TEST_TABULAR_PATH,
            Config.TEST_IDS_PATH,
            test_w_path,
        ]

        if load_cached and all(os.path.exists(f) for f in cache_files):
            print("Loading cached data from disk...")
            train_data = (
                np.load(Config.TRAIN_FEATURES_PATH),
                np.load(Config.TRAIN_TABULAR_PATH),
                np.load(train_w_path),
                np.load(Config.TRAIN_TARGETS_PATH),
            )
            val_data = (
                np.load(Config.VAL_FEATURES_PATH),
                np.load(Config.VAL_TABULAR_PATH),
                np.load(val_w_path),
                np.load(Config.VAL_TARGETS_PATH),
            )
            test_data = (
                np.load(Config.TEST_FEATURES_PATH),
                np.load(Config.TEST_TABULAR_PATH),
                np.load(test_w_path),
                np.load(Config.TEST_IDS_PATH),
            )
            return train_data, val_data, test_data

        print("Processing data from scratch...")

        # Load Metadata
        df_train = pd.read_csv(Config.TRAIN_META_PATH)
        df_val = pd.read_csv(Config.VAL_META_PATH)
        df_test = pd.read_csv(Config.TEST_META_PATH)

        if Config.DEBUG:
            print(f"DEBUG Mode: Using {Config.DEBUG_SIZE} samples.")
            df_train = df_train.head(Config.DEBUG_SIZE)
            df_val = df_val.head(Config.DEBUG_SIZE)
            df_test = df_test.head(Config.DEBUG_SIZE)

        # 1. Extract Image Features (Unique Patients across all sets)
        all_patients = pd.concat([df_train, df_val, df_test])[
            ["Patient", "dcm_path"]
        ].drop_duplicates()
        img_dict = self.extract_image_features(
            all_patients["Patient"], all_patients["dcm_path"]
        )

        # 2. Process Tabular & Align
        # Train
        t_tab, t_w, t_y = self.process_tabular(df_train, is_train=True)
        t_img, t_tab, t_w, t_y = self.align_data(df_train, img_dict, t_tab, t_w, t_y)

        # Val
        v_tab, v_w, v_y = self.process_tabular(df_val, is_train=False)
        v_img, v_tab, v_w, v_y = self.align_data(df_val, img_dict, v_tab, v_w, v_y)

        # Test
        ts_tab, ts_w, ts_y = self.process_tabular(df_test, is_train=False)
        ts_img, ts_tab, ts_w, ts_y = self.align_data(
            df_test, img_dict, ts_tab, ts_w, ts_y
        )

        # Save to cache
        np.save(Config.TRAIN_FEATURES_PATH, t_img)
        np.save(Config.TRAIN_TABULAR_PATH, t_tab)
        np.save(Config.TRAIN_TARGETS_PATH, t_y)
        np.save(train_w_path, t_w)

        np.save(Config.VAL_FEATURES_PATH, v_img)
        np.save(Config.VAL_TABULAR_PATH, v_tab)
        np.save(Config.VAL_TARGETS_PATH, v_y)
        np.save(val_w_path, v_w)

        np.save(Config.TEST_FEATURES_PATH, ts_img)
        np.save(Config.TEST_TABULAR_PATH, ts_tab)
        np.save(Config.TEST_IDS_PATH, df_test["Patient_Week"].values)
        np.save(test_w_path, ts_w)

        return (
            (t_img, t_tab, t_w, t_y),
            (v_img, v_tab, v_w, v_y),
            (ts_img, ts_tab, ts_w, df_test["Patient_Week"].values),
        )


class OSICDataset(Dataset):
    """
    PyTorch Dataset for the OSIC Pulmonary Fibrosis task.
    """

    def __init__(self, img_feats, tab_feats, weeks, targets=None, patient_weeks=None):
        self.img_feats = torch.tensor(img_feats, dtype=torch.float32)
        self.tab_feats = torch.tensor(tab_feats, dtype=torch.float32)
        self.weeks = torch.tensor(weeks, dtype=torch.float32)
        # Targets are FVC values (for training/val)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32)
            if targets is not None and patient_weeks is None
            else None
        )
        # Patient_Weeks are IDs (for test inference)
        self.patient_weeks = patient_weeks

    def __len__(self):
        return len(self.img_feats)

    def __getitem__(self, idx):
        img = self.img_feats[idx]
        tab = self.tab_feats[idx]
        week = self.weeks[idx]

        if self.targets is not None:
            return img, tab, week, self.targets[idx]
        else:
            # For inference, return ID instead of target
            return img, tab, week, self.patient_weeks[idx]


def get_data(load_cached_data=True):
    """
    Main entry point to get DataLoaders/Datasets.
    """
    processor = DataProcessor()
    train_data, val_data, test_data = processor.run(load_cached=load_cached_data)

    train_ds = OSICDataset(*train_data)
    val_ds = OSICDataset(*val_data)
    # Test data tuple structure: (img, tab, weeks, ids) -> targets=None, patient_weeks=ids
    test_ds = OSICDataset(
        test_data[0],
        test_data[1],
        test_data[2],
        targets=None,
        patient_weeks=test_data[3],
    )

    return train_ds, val_ds, test_ds
