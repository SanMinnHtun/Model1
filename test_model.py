from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from career_roadmap_model import CareerMatcherNN


PREPROCESSORS_PATH = Path("career_matcher_preprocessors.pkl")
MODEL_PATH = Path("enhanced_career_matcher_nn.pth")

CATEGORICAL_COLUMNS = [
    "programming_base",
    "preferred_task",
    "academic_strength",
    "personality_style",
]
FEATURE_COLUMNS = [
    "programming_base",
    "study_duration",
    "preferred_task",
    "academic_strength",
    "personality_style",
]
TARGET_COLUMN = "recommended_roadmap"


test_profile = {
    "programming_base": "Python",
    "study_duration": 12,
    "preferred_task": "Data_Analysis",
    "academic_strength": "Math_and_Logic",
    "personality_style": "Analytical",
}


def load_preprocessors(path: Path) -> tuple[dict[str, Any], Any, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor artifact file not found: {path}")

    with path.open("rb") as file:
        artifacts = pickle.load(file)

    encoders = artifacts.get("encoders") or artifacts.get("feature_encoders")
    target_le = artifacts.get("target_le") or artifacts.get("target_encoder")
    scaler = artifacts.get("scaler") or artifacts.get("duration_scaler")

    if encoders is None or target_le is None or scaler is None:
        raise KeyError(
            "career_matcher_preprocessors.pkl must contain 'encoders', 'target_le', and 'scaler'."
        )

    return encoders, target_le, scaler


def _safe_encode(value: Any, encoder: Any, column_name: str) -> int:
    known_classes = [str(item) for item in encoder.classes_]
    if not known_classes:
        raise ValueError(f"Encoder for '{column_name}' has no known classes.")

    candidate = str(value).strip()
    if candidate not in known_classes:
        fallback = known_classes[0]
        print(
            f"Warning: '{column_name}' value '{candidate}' is unseen. "
            f"Falling back to '{fallback}'."
        )
        candidate = fallback

    return int(encoder.transform([candidate])[0])


def build_feature_tensor(profile: dict[str, Any], encoders: dict[str, Any], scaler: Any) -> torch.FloatTensor:
    encoded_features: list[float] = []

    for column in FEATURE_COLUMNS:
        if column == "study_duration":
            raw_duration = float(profile.get(column, 0.0))
            scaled_duration = scaler.transform([[raw_duration]])[0][0]
            encoded_features.append(float(scaled_duration))
            continue

        if column not in profile:
            raise KeyError(f"Missing required feature '{column}' in test profile.")

        encoder = encoders.get(column)
        if encoder is None:
            raise KeyError(f"Missing encoder for feature '{column}'.")

        encoded_value = _safe_encode(profile[column], encoder, column)
        encoded_features.append(float(encoded_value))

    return torch.FloatTensor([encoded_features])


def load_model(input_dim: int, output_dim: int) -> CareerMatcherNN:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights file not found: {MODEL_PATH}")

    model = CareerMatcherNN(input_dim=input_dim, output_dim=output_dim)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def pretty_print_results(
    profile: dict[str, Any],
    roadmap_labels: list[str],
    probabilities: torch.Tensor,
    best_index: int,
) -> None:
    best_label = roadmap_labels[best_index]
    best_score = float(probabilities[best_index] * 100.0)

    print("\n=== Career Matcher Model Test Summary ===")
    print("Input profile:")
    for key, value in profile.items():
        print(f"  - {key}: {value}")

    print(f"\nFinal Recommended Career Roadmap: {best_label}")
    print(f"Confidence Score: {best_score:.2f}%\n")
    print("All career roadmap probabilities:")

    sorted_indices = torch.argsort(probabilities, descending=True)
    for rank, idx in enumerate(sorted_indices, start=1):
        label = roadmap_labels[int(idx)]
        score = float(probabilities[int(idx)] * 100.0)
        print(f"  {rank:02d}. {label}: {score:.2f}%")


def main() -> None:
    encoders, target_le, scaler = load_preprocessors(PREPROCESSORS_PATH)
    x_test = build_feature_tensor(test_profile, encoders, scaler)
    model = load_model(input_dim=x_test.shape[1], output_dim=len(target_le.classes_))

    with torch.no_grad():
        raw_logits = model(x_test)
        probabilities = F.softmax(raw_logits.squeeze(0), dim=0)

    roadmap_labels = [str(label) for label in target_le.classes_]
    best_index = int(torch.argmax(probabilities).item())
    pretty_print_results(test_profile, roadmap_labels, probabilities, best_index)


if __name__ == "__main__":
    main()
