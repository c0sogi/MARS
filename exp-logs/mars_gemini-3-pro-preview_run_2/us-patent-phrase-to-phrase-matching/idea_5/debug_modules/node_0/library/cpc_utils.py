import pandas as pd
from library.utils import seed_everything

# Official CPC Class Titles (Source: Cooperative Patent Classification)
# Covers common classes found in the US Patent Phrase to Phrase Matching dataset.
CPC_CODES = {
    "A01": "Agriculture; forestry; animal husbandry; hunting; trapping; fishing",
    "A21": "Baking; edible doughs",
    "A22": "Butchering; meat treatment; processing poultry or fish",
    "A23": "Foods or foodstuffs; their treatment, not covered by other classes",
    "A24": "Tobacco; cigars; cigarettes; smokers' requisites",
    "A41": "Wearing apparel",
    "A42": "Headwear",
    "A43": "Footwear",
    "A44": "Haberdashery; jewellery",
    "A45": "Hand or traveling articles",
    "A46": "Brushware",
    "A47": "Furniture; domestic articles or appliances; coffee mills; spice mills; suction cleaners in general",
    "A61": "Medical or veterinary science; hygiene",
    "A62": "Life-saving; fire-fighting",
    "A63": "Sports; games; amusements",
    "B01": "Physical or chemical processes or apparatus in general",
    "B02": "Crushing, pulverising, or disintegrating; preparatory for grain milling",
    "B03": "Separation of solid materials using liquids or using pneumatic tables or jigs; magnetic or electrostatic separation of solid materials from solid materials or fluids; separation by high-voltage electric fields",
    "B04": "Centrifugal apparatus or machines for carrying-out physical or chemical processes",
    "B05": "Spraying or atomising in general; applying liquids or other fluent materials to surfaces, in general",
    "B06": "Generating or transmitting mechanical vibrations in general",
    "B07": "Separating solids from solids; sorting",
    "B08": "Cleaning",
    "B09": "Disposal of solid waste; reclamation of contaminated soil",
    "B21": "Mechanical working of metal without essentially removing material; punching metal",
    "B22": "Casting powder metallurgy",
    "B23": "Machine tools; metal-working not otherwise provided for",
    "B24": "Grinding; polishing",
    "B25": "Hand tools; portable power-driven tools; handles for hand implements; workshop equipment; manipulators",
    "B26": "Hand cutting tools; cutting; severing",
    "B27": "Working or preserving wood or similar material; nailing or stapling machines in general",
    "B28": "Working cement, clay, or stone",
    "B29": "Working of plastics; working of substances in a plastic state in general",
    "B30": "Presses",
    "B31": "Making paper articles; working paper",
    "B32": "Layered products",
    "B33": "Additive manufacturing technology",
    "B41": "Printing; lining machines; typewriters; stamps",
    "B42": "Bookbinding; albums; files; special printed matter",
    "B43": "Writing or drawing implements; bureau accessories",
    "B44": "Decorative arts",
    "B60": "Vehicles in general",
    "B61": "Railways",
    "B62": "Land vehicles for travelling otherwise than on rails",
    "B63": "Ships or other waterborne vessels; related equipment",
    "B64": "Aircraft; aviation; cosmonautics",
    "B65": "Conveying; packing; storing; handling thin or filamentary material",
    "B66": "Hoisting; lifting; hauling",
    "B67": "Opening or closing bottles, jars or similar containers; liquid handling",
    "B68": "Saddlery; upholstery",
    "B81": "Microstructural technology",
    "B82": "Nanotechnology",
    "C01": "Inorganic chemistry",
    "C02": "Treatment of water, waste water, sewage, or sludge",
    "C03": "Glass; mineral or slag wool",
    "C04": "Cements; concrete; artificial stone; ceramics; refractories",
    "C05": "Fertilisers; manufacture thereof",
    "C06": "Explosives; matches",
    "C07": "Organic chemistry",
    "C08": "Organic macromolecular compounds; their preparation or chemical working-up; compositions based thereon",
    "C09": "Dyes; paints; polishes; natural resins; adhesives; compositions not otherwise provided for",
    "C10": "Petroleum, gas or coke industries; technical gases containing carbon monoxide; fuels; lubricants; peat",
    "C11": "Animal or vegetable oils, fats, fatty substances or waxes; fatty acids therefrom; detergents; candles",
    "C12": "Biochemistry; beer; spirits; wine; vinegar; microbiology; enzymology; mutation or genetic engineering",
    "C13": "Sugar industry",
    "C14": "Skins; hides; pelts; leather",
    "C21": "Metallurgy of iron",
    "C22": "Metallurgy; ferrous or non-ferrous alloys; treatment of alloys or non-ferrous metals",
    "C23": "Coating metallic material; coating material with metallic material; chemical surface treatment; diffusion treatment of metallic material; coating by vacuum evaporation, by sputtering, by ion implantation or by chemical vapour deposition, in general; inhibiting corrosion of metallic material or incrustation in general",
    "C25": "Electrolytic or electrophoretic processes; apparatus therefor",
    "C30": "Crystal growth",
    "C40": "Combinatorial technology",
    "D01": "Natural or artificial threads or fibres; spinning",
    "D02": "Yarns; mechanical finishing of yarns or ropes; warping or beaming",
    "D03": "Weaving",
    "D04": "Braiding; lace-making; knitting; trimmings; non-woven fabrics",
    "D05": "Sewing; embroidering; tufting",
    "D06": "Treatment of textiles or the like; laundering; flexible materials not otherwise provided for",
    "D07": "Ropes; cables other than electric",
    "D21": "Paper-making; production of cellulose",
    "E01": "Construction of roads, railways, or bridges",
    "E02": "Hydraulic engineering; foundations; soil-shifting",
    "E03": "Water supply; sewerage",
    "E04": "Building",
    "E05": "Locks; keys; window or door fittings; safes",
    "E06": "Doors, windows, shutters, or roller blinds, in general; ladders",
    "E21": "Earth or rock drilling; mining",
    "F01": "Machines or engines in general; engine plants in general; steam engines",
    "F02": "Combustion engines; hot-gas or combustion-product engine plants",
    "F03": "Machines or engines for liquids; wind, spring, or weight motors; producing mechanical power or a reactive propulsive thrust, not otherwise provided for",
    "F04": "Positive-displacement machines for liquids; pumps for liquids or elastic fluids",
    "F15": "Fluid-pressure actuators; hydraulics or pneumatics in general",
    "F16": "Engineering elements or units; general measures for producing and maintaining effective functioning of machines or installations; thermal insulation in general",
    "F17": "Storing or distributing gases or liquids",
    "F21": "Lighting",
    "F22": "Steam generation",
    "F23": "Combustion apparatus; combustion processes",
    "F24": "Heating; ranges; ventilating",
    "F25": "Refrigeration or cooling; combined heating and refrigeration systems; heat pump systems; manufacture or storage of ice; liquefaction or solidification of gases",
    "F26": "Drying",
    "F27": "Furnaces; kilns; ovens; retorts",
    "F28": "Heat exchange in general",
    "F41": "Weapons",
    "F42": "Ammunition; blasting",
    "G01": "Measuring; testing",
    "G02": "Optics",
    "G03": "Photography; cinematography; analogous techniques using waves other than optical waves; electrography; holography",
    "G04": "Horology",
    "G05": "Controlling; regulating",
    "G06": "Computing; calculating; counting",
    "G07": "Checking-devices",
    "G08": "Signalling",
    "G09": "Educating; cryptography; display; advertising; seals",
    "G10": "Musical instruments; acoustics",
    "G11": "Information storage",
    "G12": "Instrument details",
    "G16": "Information and communication technology [ICT] specially adapted for specific application fields",
    "G21": "Nuclear physics; nuclear engineering",
    "H01": "Basic electric elements",
    "H02": "Generation, conversion, or distribution of electric power",
    "H03": "Electronic circuitry",
    "H04": "Electric communication technique",
    "H05": "Electric techniques not otherwise provided for",
    "Y02": "Technologies or applications for mitigation or adaptation against climate change",
    "Y04": "Information or communication technologies having an impact on other technology areas",
    "Y10": "Technical tag schemes",
}

# Fallback for sections if specific class is not found
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


def get_cpc_text(context_code: str) -> str:
    """
    Retrieves the textual description for a given CPC context code.

    Args:
        context_code (str): The CPC code (e.g., 'A47', 'H04').

    Returns:
        str: The description of the CPC code. Returns the code itself or a section
             description if the specific class is not found in the dictionary.
    """
    if not isinstance(context_code, str):
        return str(context_code)

    # Try exact match
    if context_code in CPC_CODES:
        return CPC_CODES[context_code]

    # Try section fallback (first letter)
    if len(context_code) > 0:
        section = context_code[0]
        if section in CPC_SECTIONS:
            return f"{CPC_SECTIONS[section]} ({context_code})"

    return context_code


def map_cpc_to_text(
    df: pd.DataFrame, context_col: str = "context", output_col: str = "context_text"
) -> pd.DataFrame:
    """
    Maps a column of CPC codes in a DataFrame to their textual descriptions.

    Args:
        df (pd.DataFrame): Input DataFrame.
        context_col (str): Name of the column containing CPC codes.
        output_col (str): Name of the new column to store descriptions.

    Returns:
        pd.DataFrame: DataFrame with the new description column.
    """
    if context_col not in df.columns:
        raise ValueError(f"Column '{context_col}' not found in DataFrame.")

    df[output_col] = df[context_col].apply(get_cpc_text)
    return df
