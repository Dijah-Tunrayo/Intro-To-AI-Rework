"""Agronomic write-ups keyed by disease name.

The CNN only classifies. This table turns a class label into farmer-facing
symptoms / causes / treatment / prevention text.
"""

GENERIC = {
    "symptoms": ["Visible lesions or discoloration on the leaf surface"],
    "causes": ["Pathogen pressure combined with favourable humidity"],
    "treatment": [
        "Remove and destroy affected leaves",
        "Apply an appropriate registered fungicide or bactericide",
    ],
    "prevention": [
        "Rotate crops and avoid overhead irrigation late in the day",
        "Scout weekly and space plants for airflow",
    ],
}

HEALTHY = {
    "symptoms": ["Uniform colour, no lesions, no wilting"],
    "causes": ["No pathogen detected"],
    "treatment": ["No action needed"],
    "prevention": ["Keep monitoring weekly and maintain balanced nutrition"],
}

KNOWLEDGE: dict[str, dict[str, list[str]]] = {
    "Early blight": {
        "symptoms": [
            "Dark brown spots with concentric rings (target pattern)",
            "Yellow halo around lesions, lower leaves affected first",
        ],
        "causes": ["Alternaria solani fungus", "Warm, wet weather and leaf wetness"],
        "treatment": [
            "Remove infected lower leaves and destroy them",
            "Spray chlorothalonil or mancozeb every 7-10 days",
            "Mulch to stop soil splash onto foliage",
        ],
        "prevention": [
            "Rotate away from solanaceous crops for 2-3 years",
            "Stake plants and water at the base only",
        ],
    },
    "Late blight": {
        "symptoms": [
            "Water-soaked grey-green patches spreading fast",
            "White fuzzy growth on the leaf underside in humid mornings",
        ],
        "causes": ["Phytophthora infestans", "Cool nights with high humidity"],
        "treatment": [
            "Destroy infected plants immediately - do not compost",
            "Apply copper or metalaxyl-based fungicide to neighbouring plants",
        ],
        "prevention": [
            "Plant resistant varieties and certified seed",
            "Avoid dense canopies and improve drainage",
        ],
    },
}


def write_up(disease: str, healthy: bool) -> dict[str, list[str]]:
    if healthy:
        return HEALTHY
    for name, entry in KNOWLEDGE.items():
        if name.lower() in disease.lower():
            return entry
    return GENERIC
