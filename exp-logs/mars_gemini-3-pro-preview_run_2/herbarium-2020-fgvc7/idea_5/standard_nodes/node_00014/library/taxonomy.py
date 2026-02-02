import json
import os
import pandas as pd
from library.config import RAW_TRAIN_METADATA_PATH, TAXONOMY_MAPPING_PATH, WORKING_DIR


class TaxonomyManager:
    """
    Manages the hierarchical taxonomy mapping for the dataset.
    Maps species (category_id) to genus and family indices.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the TaxonomyManager.

        Args:
            load_cached_data (bool): If True, attempts to load from cached parquet file.
                                     If False or cache missing, processes raw metadata.
        """
        self.mapping_df = None
        self.species_to_genus = {}
        self.species_to_family = {}
        self.num_genus_classes = 0
        self.num_family_classes = 0

        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

        cache_exists = os.path.exists(TAXONOMY_MAPPING_PATH)

        if load_cached_data and cache_exists:
            try:
                self._load_from_cache()
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing raw metadata.")
                self._process_raw_metadata()
        else:
            self._process_raw_metadata()

        # Populate fast lookup dictionaries
        self._populate_lookups()

    def _process_raw_metadata(self):
        """
        Parses the raw JSON metadata, encodes genus/family, and saves to cache.
        """
        print(f"Processing raw metadata from {RAW_TRAIN_METADATA_PATH}...")

        with open(RAW_TRAIN_METADATA_PATH, "r") as f:
            data = json.load(f)

        categories = data["categories"]
        df = pd.DataFrame(categories)

        # We expect columns: 'id', 'genus', 'family', 'name'
        # 'id' is the category_id (species)

        # 1. Encode Genus
        unique_genera = sorted(df["genus"].unique())
        genus_map = {name: idx for idx, name in enumerate(unique_genera)}
        df["genus_id"] = df["genus"].map(genus_map)

        # 2. Encode Family
        unique_families = sorted(df["family"].unique())
        family_map = {name: idx for idx, name in enumerate(unique_families)}
        df["family_id"] = df["family"].map(family_map)

        # 3. Create contiguous species index
        # Sort by id to ensure deterministic mapping
        df = df.sort_values("id")
        df["species_idx"] = range(len(df))

        # 4. Prepare final mapping DataFrame
        # Index: category_id (species), Columns: genus_id, family_id, species_idx
        self.mapping_df = df[["id", "genus_id", "family_id", "species_idx"]].set_index(
            "id"
        )

        # Save to parquet
        print(f"Saving taxonomy mapping to {TAXONOMY_MAPPING_PATH}...")
        self.mapping_df.to_parquet(TAXONOMY_MAPPING_PATH)

    def _load_from_cache(self):
        """
        Loads the mapping DataFrame from the parquet cache.
        """
        print(f"Loading taxonomy mapping from {TAXONOMY_MAPPING_PATH}...")
        self.mapping_df = pd.read_parquet(TAXONOMY_MAPPING_PATH)

    def _populate_lookups(self):
        """
        Converts DataFrame to dictionaries for O(1) access.
        """
        if self.mapping_df is None:
            raise ValueError("Mapping data not initialized.")

        self.species_to_genus = self.mapping_df["genus_id"].to_dict()
        self.species_to_family = self.mapping_df["family_id"].to_dict()

        # Map raw category_id -> contiguous species_idx
        self.species_to_idx = self.mapping_df["species_idx"].to_dict()
        # Map contiguous species_idx -> raw category_id
        self.idx_to_species = {v: k for k, v in self.species_to_idx.items()}

        self.num_genus_classes = self.mapping_df["genus_id"].max() + 1
        self.num_family_classes = self.mapping_df["family_id"].max() + 1

        print(
            f"Taxonomy loaded: {len(self.species_to_genus)} species, "
            f"{self.num_genus_classes} genera, {self.num_family_classes} families."
        )

    def get_species_idx(self, raw_id):
        """Returns contiguous species index for raw category_id."""
        return self.species_to_idx.get(raw_id)

    def get_raw_id(self, idx):
        """Returns raw category_id for contiguous species index."""
        return self.idx_to_species.get(idx)

    def get_genus_id(self, species_id):
        """
        Returns the genus ID for a given species ID.
        """
        return self.species_to_genus.get(species_id)

    def get_family_id(self, species_id):
        """
        Returns the family ID for a given species ID.
        """
        return self.species_to_family.get(species_id)

    def get_num_genus(self):
        """
        Returns the total number of unique genus classes.
        """
        return int(self.num_genus_classes)

    def get_num_family(self):
        """
        Returns the total number of unique family classes.
        """
        return int(self.num_family_classes)

    def get_num_species(self):
        """
        Returns the total number of unique species classes.
        """
        return len(self.species_to_idx)
