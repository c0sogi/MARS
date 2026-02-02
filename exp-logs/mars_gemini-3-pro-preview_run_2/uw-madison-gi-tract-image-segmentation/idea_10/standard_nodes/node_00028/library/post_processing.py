import numpy as np
import pandas as pd
from scipy.ndimage import label
from library.utils import rle_decode, rle_encode
from library.config import Config


def keep_largest_component_3d(segmentation):
    """
    Refines a 3D segmentation mask by keeping only the largest connected component.
    This helps remove small floating artifacts that severely penalize the Hausdorff distance.

    Args:
        segmentation (np.ndarray): 3D binary mask of shape (Depth, Height, Width).

    Returns:
        np.ndarray: Refined 3D binary mask with only the largest component.
    """
    # Ensure input is boolean/binary
    mask = segmentation > 0

    # Label connected components
    # Default structure is connectivity=1 (6-neighbors in 3D: faces)
    labeled_mask, num_features = label(mask)

    # If no features found, return original (empty)
    if num_features == 0:
        return segmentation.astype(np.uint8)

    # Calculate size of each component
    # bincount returns count of each label value (0 to num_features)
    component_sizes = np.bincount(labeled_mask.ravel())

    # Index 0 is background, so we ignore it.
    # If len is 1, only background exists.
    if len(component_sizes) < 2:
        return np.zeros_like(segmentation, dtype=np.uint8)

    # Get the label with the maximum size (ignoring background at index 0)
    # component_sizes[1:] corresponds to labels 1..N
    # argmax gives index in sliced array, so add 1 to get actual label
    largest_label = component_sizes[1:].argmax() + 1

    # Create the refined mask
    refined_mask = (labeled_mask == largest_label).astype(np.uint8)

    return refined_mask


def process_case(case_df):
    """
    Applies 3D largest component retention to a single case's predictions.

    Args:
        case_df (pd.DataFrame): DataFrame containing predictions for one case.
                                Must contain 'class', 'slice', 'predicted', 'img_height', 'img_width'.

    Returns:
        pd.DataFrame: DataFrame with updated 'predicted' column.
    """
    # Sort by slice to ensure correct 3D volume construction along Z-axis
    case_df = case_df.sort_values("slice")

    # Get image dimensions (assuming constant for the case)
    if case_df.empty:
        return case_df

    h = int(case_df.iloc[0]["img_height"])
    w = int(case_df.iloc[0]["img_width"])

    # Process each class independently (Stomach, Large Bowel, Small Bowel)
    refined_rows = []

    for class_name in Config.CLASS_LABELS:
        class_df = case_df[case_df["class"] == class_name].copy()

        if class_df.empty:
            continue

        # Construct 3D volume
        # Shape: (Depth, Height, Width)
        num_slices = len(class_df)
        volume = np.zeros((num_slices, h, w), dtype=np.uint8)

        # Decode RLEs into the volume
        # We iterate by index to map back correctly; class_df is sorted by slice
        for i, rle in enumerate(class_df["predicted"]):
            volume[i] = rle_decode(rle, shape=(h, w))

        # Apply 3D refinement
        refined_volume = keep_largest_component_3d(volume)

        # Encode back to RLE
        new_rles = []
        for i in range(num_slices):
            mask_slice = refined_volume[i]
            # Encode
            rle = rle_encode(mask_slice)
            new_rles.append(rle)

        class_df["predicted"] = new_rles
        refined_rows.append(class_df)

    if not refined_rows:
        return case_df

    return pd.concat(refined_rows)


def refine_predictions(predictions_df, metadata_df):
    """
    Main entry point for post-processing.
    Groups predictions by case, applies 3D refinement, and returns the final dataframe.

    Args:
        predictions_df (pd.DataFrame): DataFrame with ['id', 'class', 'predicted'].
        metadata_df (pd.DataFrame): DataFrame with metadata including ['id', 'case', 'slice', 'img_width', 'img_height'].

    Returns:
        pd.DataFrame: Refined predictions DataFrame in submission format.
    """
    # 1. Merge metadata to get spatial context (Case ID, Slice ID, Dimensions)
    # We only need specific columns from metadata
    meta_cols = ["id", "case", "day", "slice", "img_width", "img_height"]

    # Metadata might contain duplicates if it's in long format (one row per class)
    # We drop duplicates on 'id' to get unique slice metadata
    meta_unique = metadata_df[meta_cols].drop_duplicates(subset=["id"])

    # Merge
    merged_df = predictions_df.merge(meta_unique, on="id", how="left")

    # Check for missing metadata
    if "case" not in merged_df.columns:
        print("Warning: Metadata merge failed. Returning original predictions.")
        return predictions_df

    # 2. Group by Case and Day and Process
    refined_results = []
    # Group by both case and day to ensure consistent dimensions within a volume
    for _, case_data in merged_df.groupby(["case", "day"]):
        refined_case_data = process_case(case_data)
        refined_results.append(refined_case_data)

    # 3. Reassemble
    if not refined_results:
        return predictions_df

    final_df = pd.concat(refined_results)

    # 4. Format for Submission
    # Ensure columns are in correct order
    return final_df[["id", "class", "predicted"]]
