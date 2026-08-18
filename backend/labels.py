"""PlantVillage-style class list used by the CNN head.

Keep this in sync with the ``classes.json`` saved next to your trained
checkpoint. If a checkpoint provides its own class list, that one wins.
"""

PLANTVILLAGE_CLASSES: list[str] = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]


def pretty(label: str) -> tuple[str, str, bool]:
    """Split a raw class label into (plant, disease, healthy)."""
    plant, _, disease = label.partition("___")
    disease = disease.replace("_", " ").strip() or "Unknown"
    healthy = disease.lower() == "healthy"
    return plant.replace("_", " "), ("Healthy" if healthy else disease), healthy
