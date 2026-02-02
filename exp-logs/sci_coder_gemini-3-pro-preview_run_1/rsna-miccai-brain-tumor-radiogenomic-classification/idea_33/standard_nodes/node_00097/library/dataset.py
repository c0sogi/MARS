import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.data_processing import RARVDataset, get_dataloaders

# Alias RARVDataset to BraTSDataset as per the task description.
# The RARVDataset class in library/data_processing.py implements the
# ROI-Anchored Relative-Volumetric logic (3 depths x 3 modalities).
BraTSDataset = RARVDataset


def get_transforms(phase: str):
    """
    Defines the spatially-preserved augmentations.

    Args:
        phase (str): One of 'train', 'valid', 'test'.

    Returns:
        A.Compose: The albumentations transform pipeline.
    """
    # Strictly Exclude: Random Translations (Shift) and Scaling to preserve ROI anchoring
    # as per the RARV strategy.
    if phase == "train":
        return A.Compose(
            [
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                ToTensorV2(),
            ]
        )
    elif phase in ["valid", "test", "val"]:
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown phase: {phase}")
