import os
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.preprocessing import preprocess_dataset


def get_transforms(phase: str):
    """
    Returns Albumentations transforms for the specific phase.
    Strictly spatial augmentations for training; normalization only for validation/test.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class LungDataset(Dataset):
    def __init__(self, df, transforms, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode

        self.patient_ids = df["Patient"].values

        # Define feature groups
        # Fusion: Contextual features for the Transformer (Static + Baseline Condition)
        # Order: Age (norm), Sex (enc), Smoke_0, Smoke_1, Smoke_2, Baseline Percent (norm)
        fusion_list = [
            "fusion_age",
            "fusion_sex",
            "fusion_smk0",
            "fusion_smk1",
            "fusion_smk2",
            "fusion_pct",
        ]
        self.fusion_data = df[fusion_list].values.astype(np.float32)

        # Anchor: Features for the Residual Head (MLP input)
        # Order: Baseline FVC (norm), Baseline Percent (norm), Week Diff (norm)
        anchor_list = ["anchor_base_fvc", "anchor_base_pct", "anchor_wd"]
        self.anchor_data = df[anchor_list].values.astype(np.float32)

        # Meta: Raw values for the explicit arithmetic trajectory: Pred = Base + Slope * Diff
        self.meta_data = df[["Baseline_FVC", "Week_Diff"]].values.astype(np.float32)

        if self.mode != "test":
            self.targets = df["FVC"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        # Construct paths to cached files
        ax_path = os.path.join(Config.CACHE_DIR, f"{pid}_axial.npy")
        cor_path = os.path.join(Config.CACHE_DIR, f"{pid}_coronal.npy")

        # Load Axial View
        if os.path.exists(ax_path):
            img_ax = np.load(ax_path)
        else:
            # Fallback: Zero tensor (should not happen if preprocessing runs correctly)
            img_ax = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        # Load Coronal View
        if os.path.exists(cor_path):
            img_cor = np.load(cor_path)
        else:
            img_cor = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        # Apply Transforms
        # Note: We apply transforms independently. Since the views are orthogonal (MIPs),
        # 2D spatial coherence between them isn't applicable in the same way as volumetric aug.
        if self.transforms:
            res_ax = self.transforms(image=img_ax)
            img_ax = res_ax["image"]

            res_cor = self.transforms(image=img_cor)
            img_cor = res_cor["image"]

        # Prepare Output Dictionary
        sample = {
            "axial": img_ax,
            "coronal": img_cor,
            "fusion": torch.tensor(self.fusion_data[idx]),
            "anchor": torch.tensor(self.anchor_data[idx]),
            "meta": torch.tensor(self.meta_data[idx]),
        }

        if self.mode != "test":
            sample["target"] = torch.tensor(self.targets[idx])

        return sample


def process_dataframe(df, mode="train", train_stats=None):
    """
    Prepares the dataframe: extracts baseline info, normalizes features, and encodes categoricals.
    Ensures consistency between Train, Val, and Test processing.
    """
    df = df.copy()

    # --- 1. Baseline Extraction & Week Diff Calculation ---
    if mode in ["train", "val"]:
        # Identify baseline visit (min Weeks) for each patient
        baseline_df = (
            df.sort_values(["Patient", "Weeks"])
            .groupby("Patient")
            .first()
            .reset_index()
        )
        baseline_df = baseline_df[["Patient", "FVC", "Percent", "Weeks"]]
        baseline_df.columns = [
            "Patient",
            "Baseline_FVC",
            "Baseline_Percent",
            "Baseline_Week",
        ]

        # Merge baseline info back to the full history
        df = pd.merge(df, baseline_df, on="Patient", how="left")

    elif mode == "test":
        # Test metadata already contains Baseline_* columns.
        # Map them to standard names if necessary or just use them.
        # Rename Predict_Week to Weeks for consistency in logic
        if "Predict_Week" in df.columns:
            df = df.rename(columns={"Predict_Week": "Weeks"})

        # Ensure Age/Sex/Smoking are available under standard names if they are named Baseline_*
        if "Baseline_Age" in df.columns:
            df["Age"] = df["Baseline_Age"]
        if "Baseline_Sex" in df.columns:
            df["Sex"] = df["Baseline_Sex"]
        if "Baseline_SmokingStatus" in df.columns:
            df["SmokingStatus"] = df["Baseline_SmokingStatus"]

    # Calculate Time Delta
    df["Week_Diff"] = df["Weeks"] - df["Baseline_Week"]

    # --- 2. Categorical Encoding ---
    # Sex: Male=0, Female=1
    df["sex_enc"] = df["Sex"].map({"Male": 0, "Female": 1})

    # SmokingStatus: One-hot encoding
    # We manually create columns to ensure all categories exist even if a split misses one
    for status in ["Ex-smoker", "Never smoker", "Currently smokes"]:
        col_name = f"smoke_{status.replace(' ', '_')}"  # e.g., smoke_Ex-smoker
        df[col_name] = (df["SmokingStatus"] == status).astype(int)

    # Rename specific one-hot columns for easy access
    df = df.rename(
        columns={
            "smoke_Ex-smoker": "smoke_0",
            "smoke_Never_smoker": "smoke_1",
            "smoke_Currently_smokes": "smoke_2",
        }
    )

    # --- 3. Normalization ---
    # Compute statistics only on Training data to avoid leakage
    if train_stats is None:
        train_stats = {
            "age_mean": df["Age"].mean(),
            "age_std": df["Age"].std(),
            "base_pct_mean": df["Baseline_Percent"].mean(),
            "base_pct_std": df["Baseline_Percent"].std(),
            "base_fvc_mean": df["Baseline_FVC"].mean(),
            "base_fvc_std": df["Baseline_FVC"].std(),
            "wd_mean": df["Week_Diff"].mean(),
            "wd_std": df["Week_Diff"].std(),
        }

    # Apply Normalization
    # Fusion Features
    df["fusion_age"] = (df["Age"] - train_stats["age_mean"]) / train_stats["age_std"]
    # Note: We use Baseline Percent for fusion to ensure the model learns from the stable prior
    df["fusion_pct"] = (
        df["Baseline_Percent"] - train_stats["base_pct_mean"]
    ) / train_stats["base_pct_std"]

    # Pass-through categoricals for Fusion
    df["fusion_sex"] = df["sex_enc"]
    df["fusion_smk0"] = df["smoke_0"]
    df["fusion_smk1"] = df["smoke_1"]
    df["fusion_smk2"] = df["smoke_2"]

    # Anchor Features
    df["anchor_base_fvc"] = (
        df["Baseline_FVC"] - train_stats["base_fvc_mean"]
    ) / train_stats["base_fvc_std"]
    df["anchor_base_pct"] = (
        df["Baseline_Percent"] - train_stats["base_pct_mean"]
    ) / train_stats["base_pct_std"]
    df["anchor_wd"] = (df["Week_Diff"] - train_stats["wd_mean"]) / train_stats["wd_std"]

    return df, train_stats


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Main entry point.
    1. Loads metadata.
    2. Triggers preprocessing (caching images).
    3. Prepares dataframes (normalization, encoding).
    4. Returns PyTorch DataLoaders.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Trigger Preprocessing
    # Combine all patients to ensure everyone is cached
    all_patients = pd.concat([train_df, val_df, test_df], ignore_index=True)
    # This function checks cache and processes only missing items
    preprocess_dataset(all_patients, load_cached_data=True)

    # 3. Process Dataframes
    # Process Train first to get statistics
    train_df, stats = process_dataframe(train_df, mode="train", train_stats=None)

    # Process Val and Test using Train statistics
    val_df, _ = process_dataframe(val_df, mode="val", train_stats=stats)
    test_df, _ = process_dataframe(test_df, mode="test", train_stats=stats)

    # 4. Create Datasets
    train_ds = LungDataset(train_df, transforms=get_transforms("train"), mode="train")
    val_ds = LungDataset(val_df, transforms=get_transforms("val"), mode="val")
    test_ds = LungDataset(test_df, transforms=get_transforms("test"), mode="test")

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
