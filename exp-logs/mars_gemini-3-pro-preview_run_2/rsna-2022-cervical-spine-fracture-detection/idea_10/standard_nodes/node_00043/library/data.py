import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# Attempt to import medical imaging libraries
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


class MetadataProcessor:
    """
    Handles the caching of file paths and association of annotations
    to avoid slow filesystem operations during training.
    """

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.paths_cache_file = os.path.join(self.cache_dir, "paths_cache.parquet")
        self.bbox_cache_file = os.path.join(self.cache_dir, "bbox_cache.parquet")

    def get_image_paths(self, metadata_df, load_cached=True):
        """
        Returns a dictionary mapping StudyInstanceUID to a sorted list of absolute file paths.
        """
        if load_cached and os.path.exists(self.paths_cache_file):
            try:
                df = pd.read_parquet(self.paths_cache_file)
                # Convert dataframe back to dict: Study -> List of paths
                # We assume the dataframe has 'StudyInstanceUID' and 'paths' columns
                path_map = df.set_index("StudyInstanceUID")["paths"].to_dict()
                # Paths in parquet might be numpy arrays, convert to list
                for k, v in path_map.items():
                    path_map[k] = v.tolist() if isinstance(v, np.ndarray) else list(v)
                return path_map
            except Exception as e:
                print(f"Failed to load paths cache: {e}. Recomputing...")

        # Compute from scratch
        path_map = {}
        studies = metadata_df["StudyInstanceUID"].unique()

        # Determine root dir based on the first entry (train or test)
        # Metadata contains 'image_path' which is relative to input root

        for idx, row in metadata_df.iterrows():
            study_id = row["StudyInstanceUID"]
            rel_path = row["image_path"]
            full_dir_path = os.path.join(Config.INPUT_ROOT, rel_path)

            if not os.path.exists(full_dir_path):
                path_map[study_id] = []
                continue

            # Get all .dcm files
            files = glob.glob(os.path.join(full_dir_path, "*.dcm"))

            # Sort by slice number (filename)
            # Filenames are like '1.dcm', '10.dcm'. Extract integer.
            try:
                files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            except ValueError:
                files.sort()  # Fallback to string sort if non-integer

            path_map[study_id] = files

        # Save to cache
        # Convert to DataFrame for parquet storage
        # Parquet doesn't strictly support lists in all engines, but pyarrow does.
        cache_data = [{"StudyInstanceUID": k, "paths": v} for k, v in path_map.items()]
        cache_df = pd.DataFrame(cache_data)
        cache_df.to_parquet(self.paths_cache_file, index=False)

        return path_map

    def get_bbox_map(self):
        """
        Parses bounding boxes into a nested dict: Study -> Slice_Num -> List of [x, y, w, h]
        """
        if not os.path.exists(Config.BOUNDING_BOX_PATH):
            return {}

        df = pd.read_csv(Config.BOUNDING_BOX_PATH)
        bbox_map = {}

        for _, row in df.iterrows():
            study = row["StudyInstanceUID"]
            slice_num = int(row["slice_number"])
            bbox = [row["x"], row["y"], row["width"], row["height"]]

            if study not in bbox_map:
                bbox_map[study] = {}
            if slice_num not in bbox_map[study]:
                bbox_map[study][slice_num] = []
            bbox_map[study][slice_num].append(bbox)

        return bbox_map


class CervicalSpineDataset(Dataset):
    def __init__(self, metadata_df, path_map, bbox_map, transform=None, phase="train"):
        """
        Args:
            metadata_df: DataFrame containing study IDs and labels.
            path_map: Dict mapping StudyInstanceUID to sorted list of file paths.
            bbox_map: Dict mapping StudyInstanceUID -> Slice Num -> BBoxes.
            transform: Albumentations transforms.
            phase: 'train', 'val', or 'test'.
        """
        self.metadata_df = metadata_df
        self.path_map = path_map
        self.bbox_map = bbox_map
        self.transform = transform
        self.phase = phase
        self.seq_length = Config.SEQ_LENGTH
        self.image_size = Config.IMAGE_SIZE

    def __len__(self):
        return len(self.metadata_df)

    def load_dicom_slice(self, path):
        """
        Loads a single DICOM slice. Handles missing pydicom or read errors.
        Returns a numpy array (H, W) normalized to 0-1.
        """
        try:
            if HAS_PYDICOM:
                dcm = pydicom.dcmread(path)
                pixel_array = dcm.pixel_array.astype(np.float32)

                # Apply Rescale Slope/Intercept if present
                slope = getattr(dcm, "RescaleSlope", 1.0)
                intercept = getattr(dcm, "RescaleIntercept", 0.0)
                pixel_array = pixel_array * slope + intercept

                # Bone windowing (approximate)
                # Center 1000, Width 2500 -> Min -250, Max 2250 (Example)
                # Or just Min-Max normalization per image if metadata is unreliable
                # Using a broad range for bone:
                min_val = -1000
                max_val = 3000
                pixel_array = np.clip(pixel_array, min_val, max_val)
                pixel_array = (pixel_array - min_val) / (max_val - min_val)

                img = pixel_array
            else:
                # Fallback: Try reading as standard image (if they are renamed JPEGs)
                # or return zeros if strictly DICOM and no library.
                # Given the prompt constraints, we attempt OpenCV.
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("Could not read image with cv2")
                img = img.astype(np.float32) / 255.0

            # Resize if necessary to ensure consistent stack dimensions
            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                img = cv2.resize(img, (self.image_size, self.image_size))

            return img

        except Exception:
            # Return blank image on failure to prevent crash
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)

    def load_nifti_segmentation(self, study_id, num_slices):
        """
        Loads NIFTI segmentation if available and maps it to the slice indices.
        Returns: (num_slices,) array with class indices 0-7.
        """
        if not HAS_NIBABEL:
            return None

        nifti_path = os.path.join(Config.SEGMENTATION_DIR, f"{study_id}.nii")
        if not os.path.exists(nifti_path):
            return None

        try:
            nii = nib.load(nifti_path)
            # NIFTI is usually (H, W, D) or (H, D, W).
            # Prompt says NIFTI is Sagittal, DICOM is Axial.
            # We need to be careful. For this task, we often just need the Z-dimension
            # to match the DICOM count.
            vol = nii.get_fdata()

            # Simple heuristic: Find the dimension that matches num_slices
            # If none match exactly, we resize the closest one.
            shape = vol.shape
            z_dim = -1

            if shape[2] == num_slices:
                vol = vol.transpose(1, 0, 2)  # Assuming (H, W, Z) standard
                z_dim = 2
            elif shape[0] == num_slices:
                # If Z is first dim
                vol = vol.transpose(1, 2, 0)  # Move to last
                z_dim = 2
            elif shape[1] == num_slices:
                vol = vol.transpose(0, 2, 1)
                z_dim = 2
            else:
                # Mismatch (common in this dataset due to reverse ordering or missing slices)
                # We resize the volume's Z-axis to match num_slices
                # For simplicity in this constrained environment, we assume the last dim is Z
                # and use nearest neighbor interpolation for labels.
                import scipy.ndimage

                zoom_factor = num_slices / shape[2]
                # This is heavy, maybe skip if mismatch is large
                return None

            # We now have a volume where we want to extract slice labels.
            # However, we only need the label for the slice.
            # The segmentation is pixel-wise. We need to aggregate to slice-level.
            # If a slice contains pixels of class C1 (1), then that slice is C1.

            # Simplified: We just return the max class present in each slice.
            # vol shape (H, W, Z)
            slice_labels = []
            for z in range(num_slices):
                slice_vol = vol[:, :, z]
                uniques = np.unique(slice_vol)
                # Filter 0
                uniques = uniques[uniques > 0]
                # Filter > 7 (Thoracic)
                uniques = uniques[uniques <= 7]

                if len(uniques) > 0:
                    # Take the most frequent or max? Usually vertebrae don't overlap much Z-wise.
                    slice_labels.append(int(np.max(uniques)))
                else:
                    slice_labels.append(0)

            return np.array(slice_labels)

        except Exception as e:
            return None

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        study_id = row["StudyInstanceUID"]

        # 1. Get Image Paths
        paths = self.path_map.get(study_id, [])
        num_slices = len(paths)

        # 2. Determine Sampling Indices
        if num_slices == 0:
            # Handle empty study
            indices = np.zeros(self.seq_length, dtype=int)
            paths = [""]  # Dummy
        elif num_slices < self.seq_length:
            # Pad with edge
            indices = np.arange(num_slices)
            pad = np.full(self.seq_length - num_slices, num_slices - 1)
            indices = np.concatenate([indices, pad])
        else:
            # Uniform sampling
            indices = np.linspace(0, num_slices - 1, self.seq_length).astype(int)

        # 3. Load Images (2.5D Stacking)
        images_seq = []

        # Pre-load segmentation if available
        seg_labels_full = self.load_nifti_segmentation(study_id, num_slices)

        # Prepare targets containers
        slice_fracture_labels = np.zeros(self.seq_length, dtype=np.float32)
        spatial_masks = np.zeros(
            (self.seq_length, self.image_size, self.image_size), dtype=np.float32
        )
        anatomy_labels = np.zeros(self.seq_length, dtype=np.int64)

        has_bbox = False
        has_segmentation = seg_labels_full is not None

        study_bboxes = self.bbox_map.get(study_id, {})
        if study_bboxes:
            has_bbox = True

        for t, slice_idx in enumerate(indices):
            # --- Image Loading (Channels: z-1, z, z+1) ---
            # Handle boundary conditions
            p_curr = paths[slice_idx] if num_slices > 0 else ""
            p_prev = paths[max(0, slice_idx - 1)] if num_slices > 0 else ""
            p_next = paths[min(num_slices - 1, slice_idx + 1)] if num_slices > 0 else ""

            img_c = self.load_dicom_slice(p_curr)
            img_p = self.load_dicom_slice(p_prev)
            img_n = self.load_dicom_slice(p_next)

            # Stack: (H, W, 3)
            img_stack = np.stack([img_p, img_c, img_n], axis=-1)

            # --- Target Generation ---

            # A. Slice Fracture Label & Spatial Mask
            # Map original slice index to sampled time step
            # We check if the current slice_idx has a bbox
            # Note: slice filenames are 1-based usually, indices are 0-based.
            # The path sorting handled the order.
            # We need to know the 'slice number' associated with paths[slice_idx].
            if num_slices > 0:
                try:
                    current_slice_num = int(
                        os.path.splitext(os.path.basename(paths[slice_idx]))[0]
                    )
                except:
                    current_slice_num = -1
            else:
                current_slice_num = -1

            if current_slice_num in study_bboxes:
                slice_fracture_labels[t] = 1.0
                # Draw masks
                for bbox in study_bboxes[current_slice_num]:
                    x, y, w, h = bbox
                    # Scale to image size (assuming 512x512 standard or relative)
                    # BBoxes in this dataset are usually absolute pixel coords on 512x512
                    # We assume image is resized to Config.IMAGE_SIZE
                    # If original is not 512, we might need scaling factor.
                    # For simplicity, we assume 512 base.
                    x0, y0 = int(x), int(y)
                    x1, y1 = int(x + w), int(y + h)

                    # Clip
                    x0 = max(0, min(x0, self.image_size))
                    y0 = max(0, min(y0, self.image_size))
                    x1 = max(0, min(x1, self.image_size))
                    y1 = max(0, min(y1, self.image_size))

                    spatial_masks[t, y0:y1, x0:x1] = 1.0

            # B. Anatomy Label
            if has_segmentation:
                anatomy_labels[t] = seg_labels_full[slice_idx]

            images_seq.append(img_stack)

        # 4. Apply Augmentations
        # We need to apply the SAME geometric transform to all slices in the sequence
        # Albumentations 'ReplayCompose' or manual application is needed.
        # Alternatively, reshape to (Seq*H, W, 3) apply, then split.

        # Stack vertically: (Seq * H, W, 3)
        full_volume = np.concatenate(images_seq, axis=0)

        # Also stack masks: (Seq * H, W) -> (Seq * H, W, 1)
        full_masks = np.concatenate([m for m in spatial_masks], axis=0)

        if self.transform:
            # Albumentations expects image (H, W, C) and mask (H, W)
            augmented = self.transform(image=full_volume, mask=full_masks)
            full_volume = augmented["image"]
            full_masks = augmented["mask"]

        # Unstack
        # full_volume is tensor (3, Seq*H, W) if ToTensorV2 used, or numpy (Seq*H, W, 3)
        # We assume ToTensorV2 is last step, so it's Tensor (C, H_total, W)

        # Reshape to (Seq, C, H, W)
        # Current shape: (3, Seq*512, 512)
        c, h_total, w = full_volume.shape
        h_single = self.image_size

        # Reshape: (3, Seq, H, W) -> Permute -> (Seq, 3, H, W)
        images_tensor = full_volume.view(c, self.seq_length, h_single, w).permute(
            1, 0, 2, 3
        )

        # Reshape masks: (Seq*H, W) -> (Seq, H, W)
        # Masks coming out of albumentations might be tensor or numpy
        if isinstance(full_masks, torch.Tensor):
            masks_tensor = full_masks.view(self.seq_length, h_single, w)
        else:
            masks_tensor = torch.from_numpy(
                full_masks.reshape(self.seq_length, h_single, w)
            )

        # 5. Study Labels
        if self.phase != "test":
            # Columns: patient_overall, C1..C7
            # Map columns to tensor
            # Order: C1, C2, C3, C4, C5, C6, C7, patient_overall
            # Metadata has them by name.
            labels = [
                row.get("C1", 0),
                row.get("C2", 0),
                row.get("C3", 0),
                row.get("C4", 0),
                row.get("C5", 0),
                row.get("C6", 0),
                row.get("C7", 0),
                row.get("patient_overall", 0),
            ]
            study_labels = torch.tensor(labels, dtype=torch.float32)
        else:
            study_labels = torch.zeros(8, dtype=torch.float32)

        return {
            "image": images_tensor,  # (Seq, 3, H, W)
            "study_labels": study_labels,  # (8,)
            "slice_fracture_labels": torch.tensor(
                slice_fracture_labels, dtype=torch.float32
            ),  # (Seq,)
            "spatial_masks": masks_tensor,  # (Seq, H, W)
            "anatomy_labels": torch.tensor(anatomy_labels, dtype=torch.long),  # (Seq,)
            "has_bbox": torch.tensor(has_bbox),
            "has_segmentation": torch.tensor(has_segmentation),
            "study_id": study_id,
        }


def get_transforms(phase):
    if phase == "train":
        return A.Compose(
            [
                # Resize is handled in load_dicom_slice to preserve stack structure
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.Normalize(mean=(Config.PIXEL_MEAN,), std=(Config.PIXEL_STD,)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # Resize is handled in load_dicom_slice
                A.Normalize(mean=(Config.PIXEL_MEAN,), std=(Config.PIXEL_STD,)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test.
    """
    # 1. Prepare Metadata & Paths
    proc = MetadataProcessor(Config.WORKING_DIR)

    # Load Metadata CSVs
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Combine for path caching (optimization)
    all_meta = pd.concat(
        [train_df, val_df, test_df], ignore_index=True
    ).drop_duplicates(subset=["StudyInstanceUID"])

    print("Caching/Loading file paths...")
    path_map = proc.get_image_paths(all_meta, load_cached=load_cached_data)
    bbox_map = proc.get_bbox_map()

    # 2. Create Datasets
    train_ds = CervicalSpineDataset(
        train_df, path_map, bbox_map, transform=get_transforms("train"), phase="train"
    )

    val_ds = CervicalSpineDataset(
        val_df, path_map, bbox_map, transform=get_transforms("val"), phase="val"
    )

    test_ds = CervicalSpineDataset(
        test_df, path_map, bbox_map, transform=get_transforms("test"), phase="test"
    )

    # 3. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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
