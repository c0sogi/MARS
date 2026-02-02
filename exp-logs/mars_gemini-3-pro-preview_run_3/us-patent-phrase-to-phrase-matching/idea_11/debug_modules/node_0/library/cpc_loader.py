import os
import glob
import pandas as pd
import numpy as np
from library.config import CFG
from library.utils import get_logger

logger = get_logger("cpc_loader.log")


class CPCLoader:
    """
    Handles loading and processing of CPC (Cooperative Patent Classification) codes.
    Maps context codes (e.g., 'A47') to hierarchical textual descriptions.
    """

    def __init__(self):
        # Hardcoded Section titles as a fallback and foundation
        # Source: https://en.wikipedia.org/wiki/Cooperative_Patent_Classification
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

    def get_cpc_texts(self, load_cached_data=True):
        """
        Generates or loads a mapping between CPC context codes and their text descriptions.

        Args:
            load_cached_data (bool): If True, attempts to load from disk cache first.

        Returns:
            pd.DataFrame: DataFrame with columns ['context', 'context_text'].
        """
        cache_path = CFG.context_map_path

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading cached context map from {cache_path}")
            try:
                df_context = pd.read_parquet(cache_path)
                return df_context
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")

        logger.info("Generating CPC context map...")

        # 2. Identify all unique contexts needed
        # We need to cover train, val, and test sets
        unique_contexts = set()

        for path in [CFG.train_path, CFG.val_path, CFG.test_path]:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if "context" in df.columns:
                    unique_contexts.update(df["context"].unique().tolist())
            else:
                logger.warning(f"Metadata file not found: {path}")

        unique_contexts = sorted(list(unique_contexts))
        logger.info(f"Found {len(unique_contexts)} unique context codes.")

        # 3. Load external CPC titles dataset if available
        # Search recursively in input directory for a file likely to be the titles dataset
        cpc_titles_map = {}
        candidate_files = glob.glob(
            os.path.join(CFG.input_dir, "**/*titles.csv"), recursive=True
        )

        if candidate_files:
            try:
                # Assume standard format: 'code', 'title'
                titles_path = candidate_files[0]
                logger.info(f"Found external CPC titles file: {titles_path}")
                df_titles = pd.read_csv(titles_path)
                # Normalize columns
                df_titles.columns = [c.lower() for c in df_titles.columns]
                if "code" in df_titles.columns and "title" in df_titles.columns:
                    cpc_titles_map = dict(zip(df_titles["code"], df_titles["title"]))
                else:
                    logger.warning(
                        "External file did not have expected 'code' and 'title' columns."
                    )
            except Exception as e:
                logger.warning(f"Error reading external CPC titles: {e}")
        else:
            logger.warning(
                "No external CPC titles file found in input. Using Section titles only."
            )

        # 4. Construct hierarchical descriptions
        data = []
        for context in unique_contexts:
            # Context is typically a Class code like "A47"
            # Hierarchy: Section (A) -> Class (A47) -> Subclass ...

            section_code = context[0] if len(context) > 0 else ""
            section_text = self.section_titles.get(section_code, "")

            # Try to get specific class description
            class_text = cpc_titles_map.get(context, "")

            parts = []
            if section_text:
                parts.append(section_text)

            if class_text:
                parts.append(class_text)
            elif not section_text:
                # Fallback if we have absolutely no text
                parts.append(f"CPC Class {context}")

            # Join with a separator that the model can distinguish, or just standard punctuation
            # Using "; " implies a continuation/refinement of topic
            full_text = "; ".join(parts)

            # Clean up text
            full_text = full_text.lower().strip()

            data.append({"context": context, "context_text": full_text})

        df_context_map = pd.DataFrame(data)

        # 5. Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_context_map.to_parquet(cache_path, index=False)
        logger.info(f"Saved context map to {cache_path}")

        return df_context_map
