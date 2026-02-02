import os
import pandas as pd
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger(os.path.join(Config.output_dir, "cpc_loader.log"))


class ContextMapper:
    """
    Handles the expansion of CPC context codes into full textual descriptions.
    Implements caching and hierarchical mapping (Section > Class > Subclass).
    """

    # CPC Section Titles (Version 2021.05)
    CPC_SECTIONS = {
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

    def __init__(self, config=Config):
        """
        Initialize the ContextMapper.

        Args:
            config: Configuration object containing paths and settings.
        """
        self.config = config
        self.map_path = config.cpc_context_map_path
        self.context_map = {}

    def fit(self, unique_contexts, load_cached_data=True):
        """
        Prepares the context mapping dictionary.
        Follows the strict caching logic: Try load -> Else compute & save.

        Args:
            unique_contexts (list or set): Collection of unique context codes (e.g., ['A47', 'H04']).
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.map_path), exist_ok=True)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(self.map_path):
            try:
                logger.info(f"Loading context map from cache: {self.map_path}")
                df_map = pd.read_parquet(self.map_path)
                # Convert dataframe back to dictionary
                self.context_map = pd.Series(
                    df_map.description.values, index=df_map.code
                ).to_dict()

                # Check if all requested contexts are covered
                missing = [c for c in unique_contexts if c not in self.context_map]
                if not missing:
                    logger.info("Cache hit: All contexts found.")
                    return
                else:
                    logger.info(
                        f"Cache partial miss: {len(missing)} contexts missing. Recomputing..."
                    )
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing from scratch.")

        # 2. Compute from scratch (if no cache, load failed, or forced reload)
        logger.info("Generating context mapping...")
        self.context_map = {}

        for code in unique_contexts:
            self.context_map[code] = self._generate_description(code)

        # 3. Save to cache
        try:
            logger.info(f"Saving context map to cache: {self.map_path}")
            # Convert dict to DataFrame for Parquet storage
            df_map = pd.DataFrame(
                list(self.context_map.items()), columns=["code", "description"]
            )
            df_map.to_parquet(self.map_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _generate_description(self, code):
        """
        Generates the hierarchical description for a given CPC code.
        Format: "Section Title [SEP] Class Code [SEP] Subclass Code"

        Since we lack the external detailed titles dataset, we use the Section Title
        and the code itself to simulate the hierarchy.
        """
        if not isinstance(code, str) or len(code) == 0:
            return ""

        section_char = code[0].upper()
        section_title = self.CPC_SECTIONS.get(section_char, "Unknown Section")

        # We construct a string that represents the hierarchy.
        # The model expects: Section > Class > Subclass.
        # We map:
        #   Section -> Section Title (e.g., "Human Necessities")
        #   Class   -> Code (e.g., "A47")
        #   Subclass-> "" (Placeholder as we don't have the specific subclass text)

        # Note: The [SEP] tokens will be handled by the tokenizer/dataset class
        # based on the model's specific separator. Here we return a unified string
        # or a format that can be easily split.
        # We will return a descriptive string: "{Section Title}; {Code}"

        return f"{section_title}; {code}"

    def get_context_text(self, code):
        """
        Retrieve the description for a context code.

        Args:
            code (str): The CPC context code.

        Returns:
            str: The expanded textual description.
        """
        return self.context_map.get(code, self._generate_description(code))
