import os
import pandas as pd
from library.config import Config


class CPCHelper:
    """
    Helper class to handle Cooperative Patent Classification (CPC) context expansion.
    Maps short CPC codes (e.g., 'A47') to textual descriptions to provide
    semantic context for the model.
    """

    def __init__(self):
        # Mapping of CPC Section codes to their official titles.
        # Source: Cooperative Patent Classification scheme.
        self.section_map = {
            "A": "Human Necessities",
            "B": "Performing Operations; Transporting",
            "C": "Chemistry; Metallurgy",
            "D": "Textiles; Paper",
            "E": "Fixed Constructions",
            "F": "Mechanical Engineering; Lighting; Heating; Weapons; Blasting",
            "G": "Physics",
            "H": "Electricity",
            "Y": "General Tagging of New Technological Developments",
        }

    def get_context_text(self, context_code: str) -> str:
        """
        Expands a CPC code into its textual description.

        Args:
            context_code (str): The CPC code (e.g., "A47", "H04W").

        Returns:
            str: The expanded textual description.
        """
        if not context_code or pd.isna(context_code):
            return ""

        code_str = str(context_code).strip()
        if not code_str:
            return ""

        # Extract the Section (first character)
        section_char = code_str[0].upper()

        # Retrieve the Section description
        # Note: In a full implementation with external data access, we would
        # look up the specific Class and Subclass descriptions here.
        # Given the constraints, we rely on the Section description.
        description = self.section_map.get(section_char, "")

        return description

    def process_dataset(
        self, df: pd.DataFrame, cache_path: str = None, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Adds a 'context_text' column to the dataframe containing the expanded context descriptions.
        Implements strict caching logic using Parquet files.

        Args:
            df (pd.DataFrame): The raw dataframe containing a 'context' column.
            cache_path (str): Path to save/load the processed parquet file.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The processed dataframe with 'context_text'.
        """
        # 1. Check Cache
        if load_cached_data and cache_path and os.path.exists(cache_path):
            try:
                print(f"Loading cached dataset from {cache_path}")
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache from {cache_path}: {e}. Recomputing...")

        # 2. Compute Data
        print("Processing dataset for context expansion...")
        df_processed = df.copy()

        if "context" not in df_processed.columns:
            raise ValueError("Dataframe must contain a 'context' column.")

        # Optimize: Get unique contexts, map them, then apply to dataframe
        unique_contexts = df_processed["context"].unique()
        context_mapping = {
            code: self.get_context_text(code) for code in unique_contexts
        }

        df_processed["context_text"] = df_processed["context"].map(context_mapping)

        # Fill any potential NaNs resulting from mapping (though get_context_text handles empty)
        df_processed["context_text"] = df_processed["context_text"].fillna("")

        # 3. Save Cache
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                print(f"Saving processed dataset to {cache_path}")
                df_processed.to_parquet(cache_path, index=False)
            except Exception as e:
                print(f"Warning: Failed to save cache to {cache_path}: {e}")

        return df_processed

    def generate_context_map(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cache_path: str = None,
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Generates a master mapping of all unique context codes found in Train, Val, and Test sets.
        Useful for vocabulary building or DAPT.
        """
        if load_cached_data and cache_path and os.path.exists(cache_path):
            try:
                print(f"Loading cached context map from {cache_path}")
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load context map cache: {e}. Recomputing...")

        print("Generating master context map...")

        # Extract unique contexts from all splits
        contexts = set()
        if "context" in train_df.columns:
            contexts.update(train_df["context"].unique())
        if "context" in val_df.columns:
            contexts.update(val_df["context"].unique())
        if "context" in test_df.columns:
            contexts.update(test_df["context"].unique())

        # Create DataFrame
        map_data = []
        for code in contexts:
            map_data.append(
                {"context": code, "context_text": self.get_context_text(code)}
            )

        df_map = pd.DataFrame(map_data)

        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df_map.to_parquet(cache_path, index=False)
            except Exception as e:
                print(f"Warning: Failed to save context map cache: {e}")

        return df_map
