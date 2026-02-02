from library.config import Config


def get_atomic_properties(element: str) -> dict:
    """
    Retrieves atomic properties for a specific element.

    This function acts as a lookup for domain-specific physical constants
    defined in the global configuration. It provides the Pauling Electronegativity,
    Shannon Ionic Radius, and Atomic Number, which are essential for constructing
    chemical disorder and electrostatic features.

    Args:
        element (str): The chemical symbol of the element (e.g., 'Al', 'Ga', 'In', 'O').

    Returns:
        dict: A dictionary containing:
            - 'EN': Pauling Electronegativity (float)
            - 'R': Shannon Ionic Radius in Angstroms (float)
            - 'Z': Atomic Number (int)

    Raises:
        ValueError: If the element is not supported or missing from the configuration.
    """
    props = Config.ATOMIC_PROPS.get(element)
    if props is None:
        raise ValueError(
            f"Atomic properties for element '{element}' are not defined in the configuration."
        )
    return props


def get_all_atomic_properties() -> dict:
    """
    Retrieves the complete dictionary of atomic properties for all supported elements.

    This is useful for iterating over all possible elements in the dataset or
    for initializing bulk feature extractors.

    Returns:
        dict: A dictionary mapping element symbols to their property dictionaries.
    """
    return Config.ATOMIC_PROPS
