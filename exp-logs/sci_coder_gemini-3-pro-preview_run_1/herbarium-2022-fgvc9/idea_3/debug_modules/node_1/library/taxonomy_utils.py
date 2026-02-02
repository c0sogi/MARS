import os
import json
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import get_class_mappings


class TaxonomyProcessor:
    """
    Handles the processing of taxonomic hierarchy (Species -> Genus -> Family).
    Creates mappings from species model indices to genus and family indices
    for hierarchical multi-task learning.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the processor and load/compute mappings.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        self.load_cached_data = load_cached_data
        self.mappings_file = os.path.join(Config.WORKING_DIR, "taxonomy_mappings.json")

        self.species_idx_to_genus_idx = []
        self.species_idx_to_family_idx = []
        self.num_genera = 0
        self.num_families = 0

        self._load_or_process()

    def _load_or_process(self):
        """
        Loads mappings from cache if available, otherwise computes them.
        """
        if self.load_cached_data and os.path.exists(self.mappings_file):
            try:
                with open(self.mappings_file, "r") as f:
                    data = json.load(f)
                    self.species_idx_to_genus_idx = data["species_idx_to_genus_idx"]
                    self.species_idx_to_family_idx = data["species_idx_to_family_idx"]
                    self.num_genera = data["num_genera"]
                    self.num_families = data["num_families"]
                return
            except Exception as e:
                print(f"Failed to load cached taxonomy mappings: {e}. Recomputing...")

        self._process_taxonomy()

    def _process_taxonomy(self):
        """
        Parses metadata to build species -> genus -> family relationships.
        """
        # 1. Get Species Mappings (aligns with model's output layer)
        class_to_idx, idx_to_class = get_class_mappings(
            load_cached_data=self.load_cached_data
        )

        # 2. Load Raw Metadata
        # We need this to look up genus/family strings for each category_id
        with open(Config.TRAIN_METADATA_JSON, "r") as f:
            raw_meta = json.load(f)

        # Create a lookup: category_id -> {genus, family}
        cat_lookup = {}
        for cat in raw_meta.get("categories", []):
            cat_id = int(cat["category_id"])
            cat_lookup[cat_id] = {
                "genus": cat.get("genus"),
                "family": cat.get("family"),
            }

        # 3. Align Taxonomy with Species Indices
        # Iterate through species indices (0 to N-1) to ensure order matches model
        ordered_genera = []
        ordered_families = []
        temp_species_data = []

        # idx_to_class keys are integers 0..15500
        for idx in range(len(idx_to_class)):
            cat_id = idx_to_class[idx]
            if cat_id not in cat_lookup:
                raise ValueError(
                    f"Category ID {cat_id} found in training data but not in metadata JSON."
                )

            info = cat_lookup[cat_id]
            ordered_genera.append(info["genus"])
            ordered_families.append(info["family"])
            temp_species_data.append(info)

        # 4. Create Integer Encodings for Genus and Family
        # Sort to ensure deterministic indexing
        unique_genera = sorted(list(set(ordered_genera)))
        unique_families = sorted(list(set(ordered_families)))

        genus_to_idx = {g: i for i, g in enumerate(unique_genera)}
        family_to_idx = {f: i for i, f in enumerate(unique_families)}

        self.num_genera = len(unique_genera)
        self.num_families = len(unique_families)

        # 5. Build Mapping Arrays
        # Index i corresponds to species_idx i
        self.species_idx_to_genus_idx = [
            genus_to_idx[d["genus"]] for d in temp_species_data
        ]
        self.species_idx_to_family_idx = [
            family_to_idx[d["family"]] for d in temp_species_data
        ]

        # 6. Save to Cache
        data_to_save = {
            "num_genera": self.num_genera,
            "num_families": self.num_families,
            "species_idx_to_genus_idx": self.species_idx_to_genus_idx,
            "species_idx_to_family_idx": self.species_idx_to_family_idx,
            "genus_to_idx": genus_to_idx,
            "family_to_idx": family_to_idx,
        }

        os.makedirs(os.path.dirname(self.mappings_file), exist_ok=True)
        with open(self.mappings_file, "w") as f:
            json.dump(data_to_save, f)

    def get_maps(self):
        """
        Returns the mappings required for the Dataset.

        Returns:
            species_to_genus (np.array): Array where index is species_idx, value is genus_idx.
            species_to_family (np.array): Array where index is species_idx, value is family_idx.
        """
        return np.array(self.species_idx_to_genus_idx), np.array(
            self.species_idx_to_family_idx
        )

    def get_counts(self):
        """
        Returns the number of unique genera and families.

        Returns:
            num_genera (int): Total number of unique genera.
            num_families (int): Total number of unique families.
        """
        return self.num_genera, self.num_families
