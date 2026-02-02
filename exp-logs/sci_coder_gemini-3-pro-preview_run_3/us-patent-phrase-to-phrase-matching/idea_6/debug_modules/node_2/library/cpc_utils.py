import os
import pandas as pd
import logging
from library.config import cfg
from library.utils import get_logger, seed_everything

# =============================================================================
# Hardcoded CPC Data (Sections and Common Classes)
# =============================================================================
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

# A subset of common CPC Classes found in patent datasets.
# In a real production environment, this would be loaded from a complete external database.
CPC_CLASSES = {
    "A01": "Agriculture; Forestry; Animal Husbandry; Hunting; Trapping; Fishing",
    "A21": "Baking; Edible Doughs",
    "A22": "Butchering; Meat Treatment; Processing Poultry or Fish",
    "A23": "Foods or Foodstuffs; Their Treatment, Not Covered by Other Classes",
    "A24": "Tobacco; Cigars; Cigarettes; Smokers' Requisites",
    "A41": "Wearing Apparel",
    "A42": "Headwear",
    "A43": "Footwear",
    "A44": "Haberdashery; Jewellery",
    "A45": "Hand or Traveling Articles",
    "A46": "Brushware",
    "A47": "Furniture; Domestic Articles or Appliances; Coffee Mills; Spice Mills; Suction Cleaners in General",
    "A61": "Medical or Veterinary Science; Hygiene",
    "A62": "Life-Saving; Fire-Fighting",
    "A63": "Sports; Games; Amusements",
    "B01": "Physical or Chemical Processes or Apparatus in General",
    "B02": "Crushing, Pulverizing, or Disintegrating; Preparatory for Grain Milling",
    "B03": "Separation of Solid Materials Using Liquids or Using Pneumatic Tables or Jigs",
    "B04": "Centrifugal Apparatus or Machines for Carrying Out Physical or Chemical Processes",
    "B05": "Spraying or Atomizing in General; Applying Liquids or Other Fluents to Surfaces",
    "B06": "Generating or Transmitting Mechanical Vibrations in General",
    "B07": "Separating Solids from Solids; Sorting",
    "B08": "Cleaning",
    "B09": "Disposal of Solid Waste; Reclamation of Contaminated Soil",
    "B21": "Mechanical Metal-Working Without Essentially Removing Material; Punching Metal",
    "B22": "Casting; Powder Metallurgy",
    "B23": "Machine Tools; Metal-Working Not Otherwise Provided For",
    "B24": "Grinding; Polishing",
    "B25": "Hand Tools; Portable Power-Driven Tools; Handles for Hand Implements; Workshop Equipment; Manipulators",
    "B26": "Hand Cutting Tools; Cutting; Severing",
    "B27": "Working or Preserving Wood or Similar Material; Nailing or Stapling Machines in General",
    "B28": "Working Cement, Clay, or Stone",
    "B29": "Working of Plastics; Working of Substances in a Plastic State in General",
    "B30": "Presses",
    "B31": "Making Paper Articles; Working Paper",
    "B32": "Layered Products",
    "B33": "Additive Manufacturing Technology",
    "B41": "Printing; Lining Machines; Typewriters; Stamps",
    "B42": "Bookbinding; Albums; Files; Special Printed Matter",
    "B43": "Writing or Drawing Implements; Bureau Accessories",
    "B44": "Decorative Arts",
    "B60": "Vehicles in General",
    "B61": "Railways",
    "B62": "Land Vehicles for Travelling Otherwise Than on Rails",
    "B63": "Ships or Other Waterborne Vessels; Related Equipment",
    "B64": "Aircraft; Aviation; Cosmonautics",
    "B65": "Conveying; Packing; Storing; Handling Thin or Filamentary Material",
    "B66": "Hoisting; Lifting; Hauling",
    "B67": "Opening or Closing Bottles, Jars or Similar Containers; Liquid Handling",
    "B68": "Saddlery; Upholstery",
    "C01": "Inorganic Chemistry",
    "C02": "Treatment of Water, Waste Water, Sewage, or Sludge",
    "C03": "Glass; Mineral or Slag Wool",
    "C04": "Cements; Concrete; Artificial Stone; Ceramics; Refractories",
    "C05": "Fertilizers; Manufacture Thereof",
    "C06": "Explosives; Matches",
    "C07": "Organic Chemistry",
    "C08": "Organic Macromolecular Compounds; Their Preparation or Chemical Working-Up; Compositions Based Thereon",
    "C09": "Dyes; Paints; Polishes; Natural Resins; Adhesives; Compositions not Otherwise Provided For",
    "C10": "Petroleum, Gas or Coke Industries; Technical Gases Containing Carbon Monoxide; Fuels; Lubricants; Peat",
    "C11": "Animal or Vegetable Oils, Fats, Fatty Substances or Waxes; Fatty Acids Therefrom; Detergents; Candles",
    "C12": "Biochemistry; Beer; Spirits; Wine; Vinegar; Microbiology; Enzymology; Mutation or Genetic Engineering",
    "C13": "Sugar Industry",
    "C14": "Skins; Hides; Pelts; Leather",
    "C21": "Metallurgy of Iron",
    "C22": "Metallurgy; Ferrous or Non-Ferrous Alloys; Treatment of Alloys or Non-Ferrous Metals",
    "C23": "Coating Metallic Material; Coating Material with Metallic Material; Chemical Surface Treatment",
    "C25": "Electrolytic or Electrophoretic Processes; Apparatus Therefor",
    "C30": "Crystal Growth",
    "C40": "Combinatorial Technology",
    "D01": "Natural or Artificial Threads or Fibres; Spinning",
    "D02": "Yarns; Mechanical Finishing of Yarns or Ropes; Warping or Beaming",
    "D03": "Weaving",
    "D04": "Braiding; Lace-Making; Knitting; Trimmings; Non-Woven Fabrics",
    "D05": "Sewing; Embroidering; Tufting",
    "D06": "Treatment of Textiles or the Like; Laundering; Flexible Materials Not Otherwise Provided For",
    "D07": "Ropes; Cables Other Than Electric",
    "D21": "Paper-Making; Production of Cellulose",
    "E01": "Construction of Roads, Railways, or Bridges",
    "E02": "Hydraulic Engineering; Foundations; Soil-Shifting",
    "E03": "Water Supply; Sewerage",
    "E04": "Building",
    "E05": "Locks; Keys; Window or Door Fittings; Safes",
    "E06": "Doors, Windows, Shutters, or Roller Blinds, in General; Ladders",
    "E21": "Earth or Rock Drilling; Mining",
    "F01": "Machines or Engines in General; Engine Plants in General; Steam Engines",
    "F02": "Combustion Engines; Hot-Gas or Combustion-Product Engine Plants",
    "F03": "Machines or Engines for Liquids; Wind, Spring, or Weight Motors; Producing Mechanical Power",
    "F04": "Positive-Displacement Machines for Liquids; Pumps for Liquids or Elastic Fluids",
    "F15": "Fluid-Pressure Actuators; Hydraulics or Pneumatics in General",
    "F16": "Engineering Elements or Units; General Measures for Producing and Maintaining Effective Functioning",
    "F17": "Storing or Distributing Gases or Liquids",
    "F21": "Lighting",
    "F22": "Steam Generation",
    "F23": "Combustion Apparatus; Combustion Processes",
    "F24": "Heating; Ranges; Ventilating",
    "F25": "Refrigeration or Cooling; Combined Heating and Refrigeration Systems; Heat Pump Systems",
    "F26": "Drying",
    "F27": "Furnaces; Kilns; Ovens; Retorts",
    "F28": "Heat Exchange in General",
    "F41": "Weapons",
    "F42": "Ammunition; Blasting",
    "G01": "Measuring; Testing",
    "G02": "Optics",
    "G03": "Photography; Cinematography; Analogous Techniques Using Waves Other Than Optical Waves",
    "G04": "Horology",
    "G05": "Controlling; Regulating",
    "G06": "Computing; Calculating; Counting",
    "G07": "Checking-Devices",
    "G08": "Signalling",
    "G09": "Educating; Cryptography; Display; Advertising; Seals",
    "G10": "Musical Instruments; Acoustics",
    "G11": "Information Storage",
    "G12": "Instrument Details",
    "G16": "Information and Communication Technology (ICT) Specially Adapted for Specific Application Fields",
    "G21": "Nuclear Physics; Nuclear Engineering",
    "H01": "Basic Electric Elements",
    "H02": "Generation, Conversion, or Distribution of Electric Power",
    "H03": "Electronic Circuitry",
    "H04": "Electric Communication Technique",
    "H05": "Electric Techniques Not Otherwise Provided For",
}


class CPCMapper:
    """
    Handles the mapping of CPC codes to their hierarchical text descriptions.
    """

    def __init__(self):
        self.logger = get_logger(os.path.join(cfg.working_dir, "cpc_utils.log"))
        self.context_map_path = cfg.context_map_path
        seed_everything(cfg.seed)

    def _get_cpc_text(self, code):
        """
        Constructs the hierarchical text for a given CPC code.
        Format: Section Description [SEP] Class Description
        """
        code = str(code).strip()
        if not code:
            return ""

        # 1. Get Section Description
        section_char = code[0]
        section_desc = CPC_SECTIONS.get(section_char, "Unknown Section")

        # 2. Get Class Description
        # The input code is expected to be the Class (e.g., A47).
        # If it's longer (e.g. A47B), we still look up the Class part (A47).
        class_code = code[:3]
        class_desc = CPC_CLASSES.get(class_code, "Unknown Class")

        # 3. Construct Hierarchy
        # We use [SEP] as a delimiter which can be handled by the tokenizer later
        return f"{section_desc} [SEP] {class_desc}"

    def run(self, load_cached_data=True):
        """
        Generates or loads the CPC context map.

        Args:
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            pd.DataFrame: DataFrame with columns ['context', 'context_text']
        """
        # 1. Try to load from cache
        if load_cached_data and os.path.exists(self.context_map_path):
            self.logger.info(f"Loading cached context map from {self.context_map_path}")
            try:
                df_map = pd.read_parquet(self.context_map_path)
                return df_map
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Regenerating...")

        self.logger.info("Generating CPC context map from scratch...")

        # 2. Load Metadata to find all unique contexts
        train_path = os.path.join(cfg.metadata_dir, "train.csv")
        test_path = os.path.join(cfg.metadata_dir, "test.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(
                "Metadata files not found. Ensure metadata generation is complete."
            )

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        # Get unique contexts
        unique_contexts = pd.concat([df_train["context"], df_test["context"]]).unique()
        self.logger.info(f"Found {len(unique_contexts)} unique context codes.")

        # 3. Create Mapping
        data = []
        for code in unique_contexts:
            text = self._get_cpc_text(code)
            data.append({"context": code, "context_text": text})

        df_map = pd.DataFrame(data)

        # 4. Save to Cache
        self.logger.info(f"Saving context map to {self.context_map_path}")
        df_map.to_parquet(self.context_map_path, index=False)

        return df_map
