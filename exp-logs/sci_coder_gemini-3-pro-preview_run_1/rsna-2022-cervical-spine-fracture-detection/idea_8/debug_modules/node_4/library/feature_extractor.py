import os
import torch
import pandas as pd
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.models import UNetLocalizer, DetailEncoder
from library.engine import generate_stage1_results, extract_features


def load_stage1_model(device=Config.DEVICE, checkpoint_path=Config.STAGE1_CHECKPOINT):
    """
    Initializes and loads the Stage 1 UNetLocalizer model.

    Args:
        device (str): The device to load the model onto.
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        nn.Module: The loaded model in evaluation mode.
    """
    model = UNetLocalizer(
        in_channels=Config.STAGE1_IN_CHANNELS, num_classes=Config.STAGE1_NUM_CLASSES
    ).to(device)

    if os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path, device=device)
    else:
        print(
            f"Warning: Stage 1 checkpoint not found at {checkpoint_path}. Initializing with random weights."
        )

    model.eval()
    return model


def load_stage2_model(device=Config.DEVICE, checkpoint_path=Config.STAGE2_CHECKPOINT):
    """
    Initializes and loads the Stage 2 DetailEncoder model.

    Args:
        device (str): The device to load the model onto.
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        nn.Module: The loaded model in evaluation mode.
    """
    model = DetailEncoder(in_channels=Config.STAGE2_IN_CHANNELS).to(device)

    if os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path, device=device)
    else:
        print(
            f"Warning: Stage 2 checkpoint not found at {checkpoint_path}. Initializing with random weights."
        )

    model.eval()
    return model


def generate_localization_metadata(
    metadata_df, model, device=Config.DEVICE, load_cached_data=True
):
    """
    Runs the Stage 1 UNetLocalizer over the dataset to generate and cache
    ROI coordinates, soft anatomical maps, and binary masks.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing StudyInstanceUIDs to process.
                                    Ideally contains all train/val/test studies to ensure
                                    the single cache file covers everything.
        model (nn.Module): Loaded Stage 1 UNetLocalizer model.
        device (str): Computation device.
        load_cached_data (bool): If True, attempts to load results from
                                 'stage1_inference_results.parquet' before running inference.

    Returns:
        pd.DataFrame: DataFrame containing localization metadata (ROI coords, maps, etc.).
    """
    seed_everything(Config.SEED)
    print("Running Stage 1: Localization Metadata Generation...")

    # Delegate to the engine function which handles batch processing and caching
    results_df = generate_stage1_results(
        metadata_df, model, device, load_cached_data=load_cached_data
    )

    return results_df


def extract_visual_features(
    metadata_df, localization_df, model, device=Config.DEVICE, load_cached_data=True
):
    """
    Runs the Stage 2 DetailEncoder to extract visual feature vectors from image crops
    defined by the localization metadata.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing StudyInstanceUIDs to process.
        localization_df (pd.DataFrame): Output from generate_localization_metadata containing
                                        ROI coordinates and anatomical maps.
        model (nn.Module): Loaded Stage 2 DetailEncoder model.
        device (str): Computation device.
        load_cached_data (bool): If True, checks if .npy feature files already exist
                                 for the requested studies.

    Returns:
        str: Path to the directory containing the extracted feature .npy files.
    """
    seed_everything(Config.SEED)
    print("Running Stage 2: Visual Feature Extraction...")

    # Delegate to the engine function which handles cropping, inference, and saving .npy files
    feature_dir = extract_features(
        metadata_df, localization_df, model, device, load_cached_data=load_cached_data
    )

    return feature_dir
