"""celldx: probabilistic per-cell battery fault diagnosis and action engine.

Importing the package never touches the network. Datasets are downloaded only by
``celldx data fetch <name>``.
"""

__version__ = "3.0.0"

CAUSES: tuple[str, ...] = (
    "healthy",
    "temperature_effect",
    "cooling_fault",
    "lithium_plating",
    "internal_damage",
    "mechanical_weld_or_tab",
    "mechanical_contact_corrosion",
    "sei_growth",
    "ambiguous_ageing",
)
NON_FAULT_CAUSES: tuple[str, ...] = ("healthy", "temperature_effect")

CAUSE_LABELS: dict[str, str] = {
    "healthy": "Healthy",
    "temperature_effect": "Temperature effect (cold cell)",
    "cooling_fault": "Cooling fault (self-heating)",
    "lithium_plating": "Lithium plating",
    "internal_damage": "Internal damage",
    "mechanical_weld_or_tab": "Mechanical fault: loose weld or tab",
    "mechanical_contact_corrosion": "Mechanical fault: contact corrosion",
    "sei_growth": "SEI growth",
    "ambiguous_ageing": "Ambiguous ageing",
}
