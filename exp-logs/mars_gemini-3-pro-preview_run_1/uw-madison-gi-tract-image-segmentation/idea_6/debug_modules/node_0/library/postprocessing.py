import numpy as np
from scipy.ndimage import label
from library.config import Config


def keep_largest_component_3d(segmentation, min_size=Config.MIN_COMPONENT_SIZE):
    """
    Applies 3D Connected Component Analysis (CCA) to a binary segmentation mask.
    Retains only the largest connected component to remove noise and improve
    Hausdorff distance scores.

    This function assumes the input is a single-channel 3D volume (e.g., one class).
    If the largest component is smaller than `min_size`, an empty mask is returned.

    Args:
        segmentation (np.ndarray): 3D binary mask (integer or boolean).
                                   Expected shape: (Depth, Height, Width).
        min_size (int): Minimum volume (in voxels) required to keep the largest component.
                        Defaults to Config.MIN_COMPONENT_SIZE.

    Returns:
        np.ndarray: Processed 3D binary mask containing only the largest component.
                    The dtype matches the input segmentation.
    """
    # Ensure we are working with a boolean mask for labeling.
    # We treat any value > 0.5 as foreground.
    binary_mask = segmentation > 0.5

    # Label connected components.
    # structure=None defaults to a squared connectivity (connectivity=1).
    # In 3D, this means 6-connectivity (faces only).
    labeled_volume, num_features = label(binary_mask)

    # If no features are found, return an empty mask immediately.
    if num_features == 0:
        return np.zeros_like(segmentation)

    # Count the number of voxels for each label.
    # labeled_volume.ravel() flattens the 3D array to 1D for efficient counting.
    # component_sizes[i] will contain the voxel count for label i.
    component_sizes = np.bincount(labeled_volume.ravel())

    # component_sizes[0] corresponds to the background (label 0).
    # If only background exists (should be caught by num_features check, but for safety):
    if len(component_sizes) < 2:
        return np.zeros_like(segmentation)

    # Find the label with the maximum size, ignoring index 0 (background).
    # component_sizes[1:] slices off the background count.
    # argmax() returns the index within the slice, so we add 1 to get the original label ID.
    largest_component_label = component_sizes[1:].argmax() + 1
    largest_size = component_sizes[largest_component_label]

    # Check if the largest component satisfies the minimum size constraint.
    # This helps filter out cases where the model predicts only tiny noise specks.
    if largest_size < min_size:
        return np.zeros_like(segmentation)

    # Create the output mask retaining only the voxels belonging to the largest component.
    # We cast the result back to the original input dtype (e.g., uint8 or float).
    cleaned_mask = (labeled_volume == largest_component_label).astype(
        segmentation.dtype
    )

    return cleaned_mask
