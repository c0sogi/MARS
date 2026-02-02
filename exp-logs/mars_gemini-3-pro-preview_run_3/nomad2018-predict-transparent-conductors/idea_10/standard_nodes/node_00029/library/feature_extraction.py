import os
import numpy as np
import pandas as pd
from ase.io import read
from library.utils import process_dataset, get_physical_features, compute_rdf_features

# Aliases to match the target file description
extract_physical_properties = get_physical_features
compute_rdf = compute_rdf_features


def process_geometry_file(file_path):
    """
    Loads an .xyz file and combines physical and RDF descriptors into a single feature vector.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        np.ndarray: Combined feature vector (Physical + RDF).
    """
    try:
        atoms = read(file_path)
        phys = extract_physical_properties(atoms)
        rdf = compute_rdf(atoms)
        # Combine lists/arrays
        return np.concatenate([phys, rdf])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def generate_features(
    train_meta_path="./metadata/train_metadata.csv",
    val_meta_path="./metadata/val_metadata.csv",
    test_meta_path="./metadata/test_metadata.csv",
    output_dir="./working/idea_10/",
    load_cached_data=True,
):
    """
    Generates features for train, validation, and test sets using caching.

    Args:
        train_meta_path (str): Path to training metadata CSV.
        val_meta_path (str): Path to validation metadata CSV.
        test_meta_path (str): Path to test metadata CSV.
        output_dir (str): Directory to store cached parquet files.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (df_train, df_val, df_test) DataFrames containing features and targets.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define cache paths
    train_cache = os.path.join(output_dir, "train_features.parquet")
    val_cache = os.path.join(output_dir, "val_features.parquet")
    test_cache = os.path.join(output_dir, "test_features.parquet")

    # Process datasets using the utility function which handles caching logic
    print("Generating Training Features...")
    df_train = process_dataset(
        train_meta_path,
        train_cache,
        input_dir="./input",
        load_cached_data=load_cached_data,
    )

    print("Generating Validation Features...")
    df_val = process_dataset(
        val_meta_path, val_cache, input_dir="./input", load_cached_data=load_cached_data
    )

    print("Generating Test Features...")
    df_test = process_dataset(
        test_meta_path,
        test_cache,
        input_dir="./input",
        load_cached_data=load_cached_data,
    )

    return df_train, df_val, df_test
