import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_DIR,
    SEED,
    N_PCA_COMPONENTS,
    BATCH_SIZE,
)
from library.image_utils import process_patient

# Ensure reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_model()

    def _init_model(self):
        # Load EfficientNet B0 pre-trained on ImageNet
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=weights)

        # Remove the classifier head to get raw features (1280 dim)
        self.model.classifier = nn.Identity()

        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def extract(self, img_tensor):
        """
        Extracts features from the 3 stratified slices.
        Args:
            img_tensor (np.ndarray): Shape (3, 224, 224), range [0, 1]
        Returns:
            features (np.ndarray): Flattened feature vector of shape (3 * 1280,)
        """
        with torch.no_grad():
            # Convert to tensor
            x = torch.from_numpy(img_tensor).float()

            # Prepare batch of 3 images
            # Input is (3, H, W) where 0=Top, 1=Mid, 2=Base
            # We treat each slice as a separate 3-channel image (grayscale replicated)
            batch = []
            for i in range(3):
                slc = x[i].unsqueeze(0)  # (1, H, W)
                slc = slc.repeat(3, 1, 1)  # (3, H, W) - Replicate to RGB
                batch.append(slc)

            # Stack into batch: (3, 3, 224, 224)
            batch_tensor = torch.stack(batch).to(self.device)

            # Normalize
            batch_tensor = self.normalize(batch_tensor)

            # Forward pass
            # Output shape: (3, 1280)
            feats = self.model(batch_tensor)

            # Flatten to single vector: [Top_Feats, Mid_Feats, Base_Feats]
            return feats.cpu().numpy().flatten()


class FeaturePipeline:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.pca = PCA(n_components=N_PCA_COMPONENTS, random_state=SEED)
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        # Columns definition
        self.num_cols = ["Age", "Baseline_FVC", "Baseline_Percent"]
        self.cat_cols = ["Sex", "SmokingStatus"]

    def _add_baseline_cols(self, df):
        """
        Identifies the baseline visit (closest to Week 0) for each patient
        and propagates its FVC, Percent, and Weeks as baseline features.
        """
        df_out = df.copy()
        baselines = []

        for pid, group in df.groupby("Patient"):
            # Find row closest to week 0
            # Ideally week 0, otherwise min absolute week
            if 0 in group["Weeks"].values:
                base_row = group[group["Weeks"] == 0].iloc[0]
            else:
                idx = group["Weeks"].abs().idxmin()
                base_row = group.loc[idx]

            baselines.append(
                {
                    "Patient": pid,
                    "Baseline_Weeks": base_row["Weeks"],
                    "Baseline_FVC": base_row["FVC"],
                    "Baseline_Percent": base_row["Percent"],
                }
            )

        df_base = pd.DataFrame(baselines)
        # Merge baseline info back to original dataframe
        df_out = pd.merge(df_out, df_base, on="Patient", how="left")
        return df_out

    def process_dataset(self, load_cached_data=True):
        """
        Main execution method. Checks cache, processes data, saves cache.
        """
        # Define cache filenames
        cache_files = {
            "train": [
                "X_fvc_train.npy",
                "y_fvc_train.npy",
                "X_unc_train.npy",
                "train_meta_proc.csv",
            ],
            "val": [
                "X_fvc_val.npy",
                "y_fvc_val.npy",
                "X_unc_val.npy",
                "val_meta_proc.csv",
            ],
            "test": [
                "X_fvc_test.npy",
                "X_unc_test.npy",
                "test_meta_proc.csv",
            ],  # No y for test
        }

        # Check if all cache files exist
        all_cached = True
        for group in cache_files.values():
            for f in group:
                if not os.path.exists(os.path.join(CACHE_DIR, f)):
                    all_cached = False
                    break

        if load_cached_data and all_cached:
            print("Loading processed features from cache...")
            return self._load_cache(cache_files)

        print("Processing dataset from scratch...")

        # 1. Load Metadata
        df_train = pd.read_csv(TRAIN_METADATA_PATH)
        df_val = pd.read_csv(VAL_METADATA_PATH)
        df_test = pd.read_csv(TEST_METADATA_PATH)

        # 2. Align Metadata (Add Baseline columns to Train/Val)
        # Test metadata already has these columns from test.csv
        df_train = self._add_baseline_cols(df_train)
        df_val = self._add_baseline_cols(df_val)

        # 3. Extract Image Features (Deep + Volumetric)
        # Get unique patients across all sets to avoid re-processing
        all_patients = pd.concat(
            [
                df_train[["Patient", "dcm_path"]],
                df_val[["Patient", "dcm_path"]],
                df_test[["Patient", "dcm_path"]],
            ]
        ).drop_duplicates(subset="Patient")

        patient_raw_feats = {}

        print(f"Extracting visual features for {len(all_patients)} patients...")
        for _, row in all_patients.iterrows():
            pid = row["Patient"]
            path = row["dcm_path"]

            # Get image tensor and volumetrics
            img, vol = process_patient(pid, path, load_cached_data=load_cached_data)

            # Extract deep features
            deep = self.extractor.extract(img)

            # Concatenate: [Deep (3840), Volume (1), Density (1)]
            combined = np.concatenate([deep, vol])
            patient_raw_feats[pid] = combined

        # 4. Dimensionality Reduction (PCA)
        # Gather training samples for PCA fitting
        train_pids = df_train["Patient"].unique()
        X_pca_train_raw = np.stack([patient_raw_feats[p] for p in train_pids])

        print(f"Fitting PCA on {len(X_pca_train_raw)} training samples...")
        self.pca.fit(X_pca_train_raw)

        # Transform all patients
        patient_pca_feats = {}
        for pid, raw in patient_raw_feats.items():
            # Transform and flatten
            patient_pca_feats[pid] = self.pca.transform(raw.reshape(1, -1))[0]

        # 5. Tabular Processing
        # Fit on Train
        self.scaler.fit(df_train[self.num_cols])
        self.ohe.fit(df_train[self.cat_cols])

        # 6. Construct Final Datasets
        results = {}

        for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            # A. Tabular Features
            X_num = self.scaler.transform(df[self.num_cols])
            X_cat = self.ohe.transform(df[self.cat_cols])
            X_tab = np.hstack([X_num, X_cat])

            # B. Image Features (PCA)
            # Map patient ID to PCA features for each row
            X_img = np.stack([patient_pca_feats[p] for p in df["Patient"]])

            # C. Base Vector: [PCA_Features, Tabular_Features]
            X_base = np.hstack([X_img, X_tab])

            # D. Time Features
            weeks = df["Weeks"].values.reshape(-1, 1)
            baseline_weeks = df["Baseline_Weeks"].values.reshape(-1, 1)

            # --- FVC Model Features ---
            # Interaction: PCA Features * Weeks
            # Broadcast weeks across PCA dimensions
            X_interaction = X_img * weeks

            # Full FVC Input: [Base, Weeks, Interaction]
            X_fvc = np.hstack([X_base, weeks, X_interaction])

            # --- Uncertainty Model Features ---
            # Horizon: |Weeks - Baseline_Weeks|
            horizon = np.abs(weeks - baseline_weeks)

            # Full Uncertainty Input: [Base, Horizon]
            X_unc = np.hstack([X_base, horizon])

            # E. Targets
            y_fvc = None
            if "FVC" in df.columns and name != "test":
                y_fvc = df["FVC"].values

            results[name] = (X_fvc, y_fvc, X_unc, df)

        # 7. Save to Cache
        self._save_cache(results, cache_files)

        return results["train"], results["val"], results["test"]

    def _save_cache(self, results, cache_files):
        os.makedirs(CACHE_DIR, exist_ok=True)
        for name, (X_fvc, y_fvc, X_unc, df) in results.items():
            files = cache_files[name]
            # Save X_fvc
            np.save(os.path.join(CACHE_DIR, files[0]), X_fvc)
            # Save y_fvc (if exists)
            if y_fvc is not None:
                np.save(os.path.join(CACHE_DIR, files[1]), y_fvc)
            # Save X_unc
            # Note: Index depends on whether y exists.
            # For train/val: files[2] is X_unc. For test: files[1] is X_unc.
            unc_idx = 2 if y_fvc is not None else 1
            np.save(os.path.join(CACHE_DIR, files[unc_idx]), X_unc)
            # Save Metadata
            meta_idx = unc_idx + 1
            df.to_csv(os.path.join(CACHE_DIR, files[meta_idx]), index=False)

    def _load_cache(self, cache_files):
        results = {}
        for name, files in cache_files.items():
            X_fvc = np.load(os.path.join(CACHE_DIR, files[0]))

            y_fvc = None
            unc_idx = 1

            # If train/val, load y
            if name != "test":
                y_fvc = np.load(os.path.join(CACHE_DIR, files[1]))
                unc_idx = 2

            X_unc = np.load(os.path.join(CACHE_DIR, files[unc_idx]))
            df = pd.read_csv(os.path.join(CACHE_DIR, files[unc_idx + 1]))

            results[name] = (X_fvc, y_fvc, X_unc, df)

        return results["train"], results["val"], results["test"]


def run_feature_pipeline(load_cached_data=True):
    """
    Helper function to instantiate and run the pipeline.
    """
    pipeline = FeaturePipeline()
    return pipeline.process_dataset(load_cached_data=load_cached_data)
