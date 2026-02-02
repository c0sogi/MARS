import os
import logging
import pandas as pd
from library.config import CFG


class CPCLoader:
    """
    Handles the loading and expansion of CPC (Cooperative Patent Classification) codes
    into textual descriptions.
    """

    def __init__(self, config=CFG):
        self.config = config
        self.logger = self._get_logger()

        # Standard CPC Section Titles
        self.section_titles = {
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

    def _get_logger(self):
        logger = logging.getLogger("cpc_loader")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
        return logger

    def get_cpc_texts(
        self, df: pd.DataFrame, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Generates or loads a mapping of CPC context codes to textual descriptions.

        Args:
            df: DataFrame containing a 'context' column with CPC codes (e.g., 'A47').
            load_cached_data: Whether to try loading from the cache first.

        Returns:
            pd.DataFrame: A DataFrame with 'context' (index or column) and 'context_text'.
        """
        cache_path = self.config.context_map_path

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(f"Loading CPC context map from cache: {cache_path}")
            try:
                context_map = pd.read_parquet(cache_path)
                return context_map
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Regenerating...")

        # 2. Generate mappings
        self.logger.info("Generating CPC context map...")

        # Extract unique contexts from the provided dataframe
        unique_contexts = df["context"].unique()
        unique_contexts.sort()

        # Check for external data source (optional enhancement if file existed)
        external_cpc_path = os.path.join(self.config.input_root, "cpc_titles.csv")
        external_data = {}
        if os.path.exists(external_cpc_path):
            try:
                cpc_df = pd.read_csv(external_cpc_path)
                # Assuming columns like 'code' and 'title' exist
                if "code" in cpc_df.columns and "title" in cpc_df.columns:
                    external_data = pd.Series(
                        cpc_df.title.values, index=cpc_df.code
                    ).to_dict()
            except Exception:
                pass

        # Generate descriptions
        data = []
        for code in unique_contexts:
            description = ""

            # Strategy 1: External Data Match
            if code in external_data:
                description = external_data[code]

            # Strategy 2: Heuristic Expansion (Section Title + Class)
            else:
                section_char = code[0]
                class_code = code[1:]

                section_title = self.section_titles.get(section_char, "Unknown Section")
                # Construct lineage: "Section Title. Class Code"
                # This provides semantic anchoring for the model
                description = f"{section_title}. Class {class_code}"

            data.append({"context": code, "context_text": description})

        context_map = pd.DataFrame(data)

        # 3. Save to cache
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            context_map.to_parquet(cache_path, index=False)
            self.logger.info(f"Saved CPC context map to cache: {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

        return context_map
