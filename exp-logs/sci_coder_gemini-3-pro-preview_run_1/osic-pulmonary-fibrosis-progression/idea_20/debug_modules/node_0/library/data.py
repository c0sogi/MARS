import os
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Import provided library functions
from library.dicom_processing import generate_orthogonal_tri_slabs
from library.utils import get_logger, seed_everything

logger = get_logger("data_module")


class LungDataset(Dataset):
    def __init__(
        self,
        mode,
        metadata_dir="./metadata",
        cache_dir="./working/idea_20",
        transform=None,
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            metadata_dir (str): Directory containing metadata csv files.
            cache_dir (str): Directory to store/load cached images.
            transform (albumentations.Compose): Optional transform to be applied on a sample.
        """
        self.mode = mode
        self.cache_dir = cache_dir
        self.transform = transform

        # Load Metadata
        file_path = os.path.join(metadata_dir, f"{mode}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Metadata file not found: {file_path}")

        self.df = pd.read_csv(file_path)

        # Preprocess / Feature Engineering
        self._prepare_data()

        # Define transforms if not provided
        if self.transform is None:
            if self.mode == "train":
                self.transform = A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.ShiftScaleRotate(
                            shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                        ),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose([ToTensorV2()])

    def _prepare_data(self):
        """
        Prepares tabular features and identifies baseline values.
        """
        # 1. Baseline Extraction
        if self.mode in ["train", "val"]:
            # For train/val, we need to find the baseline (min week) for each patient
            # and merge it back to every row.

            # Group by patient and find the row with the minimum 'Weeks'
            # We assume the earliest visit is the baseline
            baseline_df = self.df.loc[self.df.groupby("Patient")["Weeks"].idxmin()]
            baseline_df = baseline_df[
                ["Patient", "FVC", "Weeks", "Percent", "Age", "Sex", "SmokingStatus"]
            ]
            baseline_df = baseline_df.rename(
                columns={
                    "FVC": "Baseline_FVC",
                    "Weeks": "Baseline_Week",
                    "Percent": "Baseline_Percent",
                    "Age": "Baseline_Age",
                    "Sex": "Baseline_Sex",
                    "SmokingStatus": "Baseline_SmokingStatus",
                }
            )

            # Merge baseline info back to the main dataframe
            self.df = self.df.merge(baseline_df, on="Patient", how="left")

            # Target column
            self.df["Target_FVC"] = self.df["FVC"]
            self.df["Current_Week"] = self.df["Weeks"]

        elif self.mode == "test":
            # Test metadata already has Baseline_ columns and Predict_Week
            # We map them to standard names used in __getitem__
            self.df["Current_Week"] = self.df["Predict_Week"]
            # Target is placeholder in test
            self.df["Target_FVC"] = self.df["FVC"]

        # 2. Feature Normalization / Encoding
        # We hardcode normalization stats based on typical dataset statistics to avoid data leakage
        # and ensure consistency between train/test without refitting scalers.
        # Approx Stats: Age Mean~67 Std~7, Percent Mean~77 Std~20

        # Normalize Age (using Baseline Age)
        self.df["Age_norm"] = (self.df["Baseline_Age"] - 67.0) / 7.0

        # Normalize Percent (using Baseline Percent)
        self.df["Percent_norm"] = (self.df["Baseline_Percent"] - 77.0) / 20.0

        # One-Hot Encode Sex
        self.df["Sex_Male"] = (self.df["Baseline_Sex"] == "Male").astype(float)
        self.df["Sex_Female"] = (self.df["Baseline_Sex"] == "Female").astype(float)

        # One-Hot Encode SmokingStatus
        # Categories: 'Ex-smoker', 'Never smoked', 'Currently smokes'
        self.df["Smoke_Ex"] = (self.df["Baseline_SmokingStatus"] == "Ex-smoker").astype(
            float
        )
        self.df["Smoke_Never"] = (
            self.df["Baseline_SmokingStatus"] == "Never smoked"
        ).astype(float)
        self.df["Smoke_Current"] = (
            self.df["Baseline_SmokingStatus"] == "Currently smokes"
        ).astype(float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images
        # DICOM directory is relative to input root.
        # The metadata contains 'dicom_dir' like 'train/ID...' or 'test/ID...'
        # We assume the input root is './input'
        dicom_dir = os.path.join("./input", row["dicom_dir"])

        # Generate or load cached Tri-Slabs
        # This returns a dict {'axial': np.array, 'coronal': np.array}
        # Images are (224, 224, 3) float32 in [0, 1]
        images_dict = generate_orthogonal_tri_slabs(
            patient_id=patient_id, patient_dir=dicom_dir, load_cached_data=True
        )

        axial_img = images_dict["axial"]
        coronal_img = images_dict["coronal"]

        # 2. Apply Transforms
        if self.transform:
            # Albumentations expects images as keyword arguments if multiple
            # We apply the same spatial transform to both to maintain some consistency?
            # Actually, axial and coronal are orthogonal, so spatial consistency isn't strictly 1:1
            # in 2D projection space (flipping axial doesn't mean flipping coronal in the same way).
            # We treat them as independent views for augmentation purposes or apply independent augs.
            # Here we apply independent augmentations.

            aug_axial = self.transform(image=axial_img)["image"]
            aug_coronal = self.transform(image=coronal_img)["image"]
        else:
            # Fallback to tensor conversion
            aug_axial = torch.from_numpy(axial_img.transpose(2, 0, 1))
            aug_coronal = torch.from_numpy(coronal_img.transpose(2, 0, 1))

        # 3. Prepare Tabular Vector (for MLP)
        # Vector: [Age, Sex_M, Sex_F, Smoke_Ex, Smoke_Never, Smoke_Current, Percent]
        tabular_feats = np.array(
            [
                row["Age_norm"],
                row["Sex_Male"],
                row["Sex_Female"],
                row["Smoke_Ex"],
                row["Smoke_Never"],
                row["Smoke_Current"],
                row["Percent_norm"],
            ],
            dtype=np.float32,
        )

        # 4. Prepare Meta Vector (for Anchor Logic)
        # Vector: [Baseline_FVC, Baseline_Week, Current_Week]
        meta_feats = np.array(
            [row["Baseline_FVC"], row["Baseline_Week"], row["Current_Week"]],
            dtype=np.float32,
        )

        # 5. Target
        target = np.array([row["Target_FVC"]], dtype=np.float32)

        return {
            "img_axial": aug_axial,
            "img_coronal": aug_coronal,
            "tabular": torch.from_numpy(tabular_feats),
            "meta": torch.from_numpy(meta_feats),
            "target": torch.from_numpy(target),
            "patient_week": str(row.get("Patient_Week", "")),  # Helper for submission
        }


def get_dataloaders(batch_size=32, num_workers=4, metadata_dir="./metadata"):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        metadata_dir (str): Path to metadata directory.

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    seed_everything(42)

    loaders = {}
    modes = ["train", "val", "test"]

    for mode in modes:
        try:
            dataset = LungDataset(mode=mode, metadata_dir=metadata_dir)

            shuffle = mode == "train"
            drop_last = mode == "train"

            loaders[mode] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=drop_last,
            )
            logger.info(f"Initialized {mode} loader with {len(dataset)} samples.")

        except FileNotFoundError:
            logger.warning(
                f"Could not create dataset for mode '{mode}'. Metadata file missing."
            )

    return loaders
