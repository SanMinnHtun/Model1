from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from career_roadmap_model import CareerMatcherNN


PREPROCESSORS_PATH = Path("career_matcher_preprocessors.pkl")
MODEL_PATH = Path("enhanced_career_matcher_nn.pth")

FEATURE_COLUMNS = [
    "programming_base",
    "study_duration",
    "preferred_task",
    "academic_strength",
    "personality_style",
]

AVAILABLE_PROGRAMMING_BASES = ["Python", "Java", "C++", "JavaScript", "Go"]
AVAILABLE_PREFERRED_TASKS = [
    "Data_Analysis",
    "Web_Development",
    "Mobile_Apps",
    "Scripting",
    "Cloud_Architecture",
]
AVAILABLE_ACADEMIC_STRENGTHS = [
    "Math_and_Logic",
    "Creativity_and_Design",
    "Problem_Solving",
    "Memorization",
]
AVAILABLE_PERSONALITY_STYLES = [
    "Analytical",
    "Practical",
    "Theoretical",
    "Creative",
]


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
            raise KeyError(f"Missing required feature '{column}' in profile.")

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


def prompt_with_options(prompt_message: str, options: list[str]) -> str:
    print(prompt_message)
    print("Available options:")
    for option in options:
        print(f"  - {option}")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "exit":
            raise KeyboardInterrupt
        if user_input in options:
            return user_input
        print("Invalid entry. Please choose one of the available options or type 'exit' to quit.")


def prompt_study_duration() -> int:
    print("Enter study_duration in months (e.g. 6, 12). Type 'exit' to quit.")

    while True:
        value = input("> ").strip()
        if value.lower() == "exit":
            raise KeyboardInterrupt

        try:
            duration = int(value)
            if duration < 0:
                raise ValueError("Study duration must be a non-negative integer.")
            return duration
        except ValueError:
            print("Invalid number. Please enter a valid integer for study duration.")


def pretty_print_results(
    profile: dict[str, Any],
    roadmap_labels: list[str],
    probabilities: torch.Tensor,
    best_index: int,
) -> None:
    best_label = roadmap_labels[best_index]
    best_score = float(probabilities[best_index] * 100.0)
    divider = "=" * 60

    print(f"\n{divider}")
    print("Career Matcher Inference Result")
    print(f"{divider}\n")
    print("Input profile:")
    for key, value in profile.items():
        print(f"  - {key}: {value}")

    print("\n\033[1mFinal Recommended Career Roadmap:\033[0m")
    print(f"  >> {best_label}\n")
    print(f"Confidence Score: {best_score:.2f}%\n")

    print("Ranked career roadmap probabilities:")
    sorted_indices = torch.argsort(probabilities, descending=True)
    for rank, idx in enumerate(sorted_indices, start=1):
        label = roadmap_labels[int(idx)]
        score = float(probabilities[int(idx)] * 100.0)
        print(f"  {rank:02d}. {label}: {score:.2f}%")
    print(divider)


def run_interactive_session(model: CareerMatcherNN, encoders: dict[str, Any], target_le: Any, scaler: Any) -> None:
    while True:
        try:
            print("\n--- New profile test (type 'exit' at any prompt to quit) ---")
            programming_base = prompt_with_options(
                "Select your primary programming language:",
                AVAILABLE_PROGRAMMING_BASES,
            )
            preferred_task = prompt_with_options(
                "Select your preferred task:",
                AVAILABLE_PREFERRED_TASKS,
            )
            academic_strength = prompt_with_options(
                "Select your academic strength:",
                AVAILABLE_ACADEMIC_STRENGTHS,
            )
            personality_style = prompt_with_options(
                "Select your personality style:",
                AVAILABLE_PERSONALITY_STYLES,
            )
            study_duration = prompt_study_duration()

            profile = {
                "programming_base": programming_base,
                "study_duration": study_duration,
                "preferred_task": preferred_task,
                "academic_strength": academic_strength,
                "personality_style": personality_style,
            }

            x_test = build_feature_tensor(profile, encoders, scaler)
            with torch.no_grad():
                raw_logits = model(x_test)
                probabilities = F.softmax(raw_logits.squeeze(0), dim=0)

            roadmap_labels = [str(label) for label in target_le.classes_]
            best_index = int(torch.argmax(probabilities).item())
            pretty_print_results(profile, roadmap_labels, probabilities, best_index)

        except KeyboardInterrupt:
            print("\nExit requested. Goodbye.")
            break
        except Exception as exc:
            print(f"Error during inference: {exc}")
            print("Please try again or type 'exit' to quit.")


def main() -> None:
    encoders, target_le, scaler = load_preprocessors(PREPROCESSORS_PATH)
    x_dummy = torch.zeros((1, len(FEATURE_COLUMNS)), dtype=torch.float32)
    model = load_model(input_dim=x_dummy.shape[1], output_dim=len(target_le.classes_))
    run_interactive_session(model, encoders, target_le, scaler)


if __name__ == "__main__":
    main()
