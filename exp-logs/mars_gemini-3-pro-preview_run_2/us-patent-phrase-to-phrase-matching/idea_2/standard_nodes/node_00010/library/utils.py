import os
import random
import numpy as np
import torch
from scipy.stats import pearsonr


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_score(y_true, y_pred):
    """
    Computes the Pearson correlation coefficient between true and predicted scores.

    Args:
        y_true (array-like): Ground truth scores.
        y_pred (array-like): Predicted scores.

    Returns:
        float: The Pearson correlation coefficient.
    """
    # Ensure inputs are flattened 1D arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Avoid errors if inputs are empty
    if len(y_true) < 2:
        return 0.0

    score, _ = pearsonr(y_true, y_pred)
    return score


def get_cpc_mapping():
    """
    Returns a dictionary mapping CPC codes to their textual descriptions.
    Based on the Cooperative Patent Classification scheme.

    Returns:
        dict: Mapping of CPC code (str) to description (str).
    """
    return {
        # Section A: Human Necessities
        "A": "Human Necessities",
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
        # Section B: Performing Operations; Transporting
        "B": "Performing Operations; Transporting",
        "B01": "Physical or Chemical Processes or Apparatus in General",
        "B02": "Crushing, Pulverising, or Disintegrating; Preparatory for Grain Milling",
        "B03": "Separation of Solid Materials Using Liquids or Using Pneumatic Tables or Jigs; Magnetic or Electrostatic Separation of Solid Materials From Solid Materials or Fluids; Separation by High-Voltage Electric Fields",
        "B04": "Centrifugal Apparatus or Machines for Carrying-Out Physical or Chemical Processes",
        "B05": "Spraying or Atomising in General; Applying Liquids or Other Fluents to Surfaces, in General",
        "B06": "Generating or Transmitting Mechanical Vibrations in General",
        "B07": "Separating Solids From Solids; Sorting",
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
        "B81": "Micro-Structural Technology; Nano-Technology",
        "B82": "Nanotechnology",
        # Section C: Chemistry; Metallurgy
        "C": "Chemistry; Metallurgy",
        "C01": "Inorganic Chemistry",
        "C02": "Treatment of Water, Waste Water, Sewage, or Sludge",
        "C03": "Glass; Mineral or Slag Wool",
        "C04": "Cements; Concrete; Artificial Stone; Ceramics; Refractories",
        "C05": "Fertilisers; Manufacture Thereof",
        "C06": "Explosives; Matches",
        "C07": "Organic Chemistry",
        "C08": "Organic Macromolecular Compounds; Their Preparation or Chemical Working-Up; Compositions Based Thereon",
        "C09": "Dyes; Paints; Polishes; Natural Resins; Adhesives; Compositions Not Otherwise Provided For",
        "C10": "Petroleum, Gas or Coke Industries; Technical Gases Containing Carbon Monoxide; Fuels; Lubricants; Peat",
        "C11": "Animal or Vegetable Oils, Fats, Fatty Substances or Waxes; Fatty Acids Therefrom; Detergents; Candles",
        "C12": "Biochemistry; Beer; Spirits; Wine; Vinegar; Microbiology; Enzymology; Mutation or Genetic Engineering",
        "C13": "Sugar Industry",
        "C14": "Skins; Hides; Pelts; Leather",
        "C21": "Metallurgy of Iron",
        "C22": "Metallurgy; Ferrous or Non-Ferrous Alloys; Treatment of Alloys or Non-Ferrous Metals",
        "C23": "Coating Metallic Material; Coating Material with Metallic Material; Chemical Surface Treatment; Diffusion Treatment of Metallic Material; Coating by Vacuum Evaporation, by Sputtering, by Ion Implantation or by Chemical Vapour Deposition, in General; Inhibiting Corrosion of Metallic Material or Incrustation in General",
        "C25": "Electrolytic or Electrophoretic Processes; Apparatus Therefor",
        "C30": "Crystal Growth",
        "C40": "Combinatorial Technology",
        # Section D: Textiles; Paper
        "D": "Textiles; Paper",
        "D01": "Natural or Man-Made Threads or Fibres; Spinning",
        "D02": "Yarns; Mechanical Finishing of Yarns or Ropes; Warping or Beaming",
        "D03": "Weaving",
        "D04": "Braiding; Lace-Making; Knitting; Trimmings; Non-Woven Fabrics",
        "D05": "Sewing; Embroidering; Tufting",
        "D06": "Treatment of Textiles or the Like; Laundering; Flexible Materials Not Otherwise Provided For",
        "D07": "Ropes; Cables Other Than Electric",
        "D21": "Paper-Making; Production of Cellulose",
        # Section E: Fixed Constructions
        "E": "Fixed Constructions",
        "E01": "Construction of Roads, Railways, or Bridges",
        "E02": "Hydraulic Engineering; Foundations; Soil-Shifting",
        "E03": "Water Supply; Sewerage",
        "E04": "Building",
        "E05": "Locks; Keys; Window or Door Fittings; Safes",
        "E06": "Doors, Windows, Shutters, or Roller Blinds, in General; Ladders",
        "E21": "Earth or Rock Drilling; Mining",
        # Section F: Mechanical Engineering; Lighting; Heating; Weapons; Blasting
        "F": "Mechanical Engineering; Lighting; Heating; Weapons; Blasting",
        "F01": "Machines or Engines in General; Engine Plants in General; Steam Engines",
        "F02": "Combustion Engines; Hot-Gas or Combustion-Product Engine Plants",
        "F03": "Machines or Engines for Liquids; Wind, Spring, or Weight Motors; Producing Mechanical Power or a Reactive Propulsive Thrust, Not Otherwise Provided For",
        "F04": "Positive-Displacement Machines for Liquids; Pumps for Liquids or Elastic Fluids",
        "F15": "Fluid-Pressure Actuators; Hydraulics or Pneumatics in General",
        "F16": "Engineering Elements or Units; General Measures for Producing and Maintaining Effective Functioning of Machines or Installations; Thermal Insulation in General",
        "F17": "Storing or Distributing Gases or Liquids",
        "F21": "Lighting",
        "F22": "Steam Generation",
        "F23": "Combustion Apparatus; Combustion Processes",
        "F24": "Heating; Ranges; Ventilating",
        "F25": "Refrigeration or Cooling; Combined Heating and Refrigeration Systems; Heat Pump Systems; Manufacture or Storage of Ice; Liquefaction or Solidification of Gases",
        "F26": "Drying",
        "F27": "Furnaces; Kilns; Ovens; Retorts",
        "F28": "Heat Exchange in General",
        "F41": "Weapons",
        "F42": "Ammunition; Blasting",
        # Section G: Physics
        "G": "Physics",
        "G01": "Measuring; Testing",
        "G02": "Optics",
        "G03": "Photography; Cinematography; Analogous Techniques Using Waves Other Than Optical Waves; Electrography; Holography",
        "G04": "Horology",
        "G05": "Controlling; Regulating",
        "G06": "Computing; Calculating; Counting",
        "G07": "Checking-Devices",
        "G08": "Signalling",
        "G09": "Educating; Cryptography; Display; Advertising; Seals",
        "G10": "Musical Instruments; Acoustics",
        "G11": "Information Storage",
        "G12": "Instrument Details",
        "G16": "Information and Communication Technology [ICT] Specially Adapted for Specific Application Fields",
        "G21": "Nuclear Physics; Nuclear Engineering",
        # Section H: Electricity
        "H": "Electricity",
        "H01": "Basic Electric Elements",
        "H02": "Generation, Conversion, or Distribution of Electric Power",
        "H03": "Electronic Circuitry",
        "H04": "Electric Communication Technique",
        "H05": "Electric Techniques Not Otherwise Provided For",
        # Section Y: General Tagging
        "Y": "General Tagging of New Technological Developments",
        "Y02": "Technologies or Applications for Mitigation or Adaptation against Climate Change",
        "Y04": "Information or Communication Technologies for the Operation, Monitoring or Maintenance of the Electric Power Grid",
        "Y10": "Technical Subjects Covered by Former USPC",
    }
