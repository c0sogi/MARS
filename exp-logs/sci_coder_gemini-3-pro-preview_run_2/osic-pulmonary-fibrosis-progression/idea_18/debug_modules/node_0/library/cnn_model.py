import os
import numpy as np
import pandas as pd
import torch
import timm
from torchvision import transforms
from library.config import Config
from library.image_utils import process_patient


# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


class TextureExtractor:
    """
    Wrapper for EfficientNet-B0 to extract features from 2.5D slices.
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load pre-trained EfficientNet-B0
        # num_classes=0 returns the pooled features (before the classifier)
        # timm automatically handles the removal of the head
        self.model = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0
        )
        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization
        # Mean and Std for RGB channels
        self.transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )

    def extract(self, images):
        """
        Extract features from a batch of images.

        Args:
            images (np.ndarray): Shape (N, H, W, 3), values in [0, 1].

        Returns:
            np.ndarray: Shape (N, Feature_Dim)
        """
        if images.shape[0] == 0:
            return np.zeros((0, self.model.num_features), dtype=np.float32)

        # Convert to Tensor: (N, H, W, 3) -> (N, 3, H, W)
        img_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()

        # Apply Normalization
        # Note: transforms.Normalize operates on tensors (C, H, W) or (N, C, H, W)
        img_tensor = self.transform(img_tensor)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            features = self.model(img_tensor)

        return features.cpu().numpy()


def generate_embeddings(
    metadata_df, dataset_name, load_cached_data=True, sample_size=None
):
    """
    Generates CNN embeddings and retrieves radiomics for a list of patients.
    Implements caching for the aggregated dataset features.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'Patient' and 'dcm_path'.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for caching.
        load_cached_data (bool): Whether to use cached features.
        sample_size (int, optional): Limit the number of patients processed (for debugging).

    Returns:
        dict: Dictionary containing:
            - 'patient_ids': List of patient IDs
            - 'embeddings': np.ndarray (Num_Patients, Feature_Dim)
            - 'radiomics': np.ndarray (Num_Patients, Radiomics_Dim)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache file path
    cache_file = os.path.join(Config.CACHE_DIR, f"features_{dataset_name}.npy")

    # 1. Try Load Cache
    # We only load from cache if sample_size is NOT set (to avoid loading full data when debugging)
    # or if the cache was generated with the same constraints (which we can't easily verify, so we skip if debugging)
    if load_cached_data and os.path.exists(cache_file) and sample_size is None:
        try:
            print(f"Loading cached features for {dataset_name} from {cache_file}...")
            data = np.load(cache_file, allow_pickle=True).item()
            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Generating embeddings for {dataset_name}...")

    extractor = TextureExtractor()

    # Get unique patients
    unique_patients = metadata_df[["Patient", "dcm_path"]].drop_duplicates()

    # Apply sampling if requested
    if sample_size is not None:
        print(f"Debugging: Limiting to {sample_size} patients.")
        unique_patients = unique_patients.head(sample_size)

    patient_ids = []
    embeddings_list = []
    radiomics_list = []

    # Iterate over unique patients
    for _, row in unique_patients.iterrows():
        patient_id = row["Patient"]
        rel_path = row["dcm_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Process patient (get 2.5D slices and radiomics)
        # We use load_cached_data=True here to leverage the per-patient cache in image_utils
        slices, radiomics = process_patient(
            patient_id, full_path, load_cached_data=load_cached_data
        )

        # Extract Deep Features
        # slices shape: (5, 224, 224, 3)
        feats = extractor.extract(slices)  # (5, 1280)

        # Mean Pooling over the 5 slices
        # If slices are empty/dummy (zeros), features will be valid but meaningless (embedding of black image).
        # This maintains pipeline stability.
        patient_embedding = np.mean(feats, axis=0)  # (1280,)

        patient_ids.append(patient_id)
        embeddings_list.append(patient_embedding)
        radiomics_list.append(radiomics)

    # Convert to numpy arrays
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    radiomics_array = np.array(radiomics_list, dtype=np.float32)

    result = {
        "patient_ids": patient_ids,
        "embeddings": embeddings_array,
        "radiomics": radiomics_array,
    }

    # 3. Save Cache
    # Only save if we processed the full dataset (sample_size is None)
    if sample_size is None:
        try:
            np.save(cache_file, result)
            print(f"Saved features for {dataset_name} to {cache_file}")
        except Exception as e:
            print(f"Warning: Could not save cache file: {e}")

    return result
