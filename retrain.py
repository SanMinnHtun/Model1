from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from career_roadmap_model import CareerMatcherNN

DATA_PATH = Path("synthetic_data.json")
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

EPOCHS = 50
LEARNING_RATE = 0.0005


def get_artifact(artifacts: dict[str, Any], primary_key: str, *fallback_keys: str) -> Any:
    for key in (primary_key, *fallback_keys):
        if key in artifacts:
            return artifacts[key]
    expected_keys = ", ".join((primary_key, *fallback_keys))
    raise KeyError(f"Missing required preprocessor artifact. Expected one of: {expected_keys}")


def load_training_data(path: Path) -> pd.DataFrame:
    print(f"Loading training dataset from {path}...")
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Rename your JSON file to synthetic_data.json."
        )

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("synthetic_data.json must contain a JSON array of objects.")

    df = pd.DataFrame(records)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df[required_columns].copy()
    print(f"Loaded {len(df)} records.")
    return df


def load_preprocessors(path: Path) -> tuple[dict[str, Any], Any, Any]:
    print(f"Loading preprocessing artifacts from {path}...")
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor artifact file not found: {path}")

    with path.open("rb") as file:
        artifacts = pickle.load(file)

    encoders = get_artifact(artifacts, "encoders", "feature_encoders")
    target_le = get_artifact(artifacts, "target_le", "target_encoder")
    scaler = get_artifact(artifacts, "scaler", "duration_scaler")

    print("Preprocessor artifacts loaded successfully.")
    return encoders, target_le, scaler


def _safe_encode_series(values: pd.Series, encoder: Any, column_name: str) -> pd.Series:
    known_classes = [str(item) for item in encoder.classes_]
    if not known_classes:
        raise ValueError(f"Encoder for '{column_name}' has no known classes.")

    fallback_value = known_classes[0]
    raw_values = values.fillna("").astype(str)
    unseen_mask = ~raw_values.isin(known_classes)
    if unseen_mask.any():
        unseen_values = sorted(set(raw_values[unseen_mask].tolist()))
        print(
            f"Warning: Column '{column_name}' contains unseen values {unseen_values}. "
            f"Falling back to '{fallback_value}' for those entries."
        )

    safe_values = raw_values.where(~unseen_mask, fallback_value)
    return pd.Series(encoder.transform(safe_values), index=values.index)


def preprocess_dataframe(
    df: pd.DataFrame,
    encoders: dict[str, Any],
    target_le: Any,
    scaler: Any,
) -> tuple[torch.FloatTensor, torch.LongTensor]:
    print("Preprocessing categorical features, study duration, and target labels...")
    processed = df.copy()

    for column in CATEGORICAL_COLUMNS:
        if column not in encoders:
            raise KeyError(f"Missing encoder for categorical column '{column}'")

        processed[column] = _safe_encode_series(processed[column], encoders[column], column)

    try:
        processed[["study_duration"]] = scaler.transform(processed[["study_duration"]].astype(float))
    except Exception as exc:
        raise ValueError("Failed to scale the study_duration column.") from exc

    known_targets = [str(item) for item in target_le.classes_]
    if not known_targets:
        raise ValueError("Target encoder has no known classes.")

    raw_targets = processed[TARGET_COLUMN].fillna("").astype(str)
    unseen_target_mask = ~raw_targets.isin(known_targets)
    if unseen_target_mask.any():
        unseen_targets = sorted(set(raw_targets[unseen_target_mask].tolist()))
        fallback_target = known_targets[0]
        print(
            f"Warning: Target column '{TARGET_COLUMN}' contains unseen labels {unseen_targets}. "
            f"Falling back to '{fallback_target}' for those entries."
        )
        raw_targets = raw_targets.where(~unseen_target_mask, fallback_target)

    features = processed[FEATURE_COLUMNS].to_numpy(dtype="float32")
    targets = target_le.transform(raw_targets)

    x_tensor = torch.FloatTensor(features)
    y_tensor = torch.LongTensor(targets)
    print(f"Prepared tensors: X={tuple(x_tensor.shape)}, y={tuple(y_tensor.shape)}")
    return x_tensor, y_tensor


def load_model(input_dim: int, output_dim: int) -> CareerMatcherNN:
    print(f"Initializing CareerMatcherNN with input_dim={input_dim}, output_dim={output_dim}...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights file not found: {MODEL_PATH}")

    model = CareerMatcherNN(input_dim=input_dim, output_dim=output_dim)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.train()
    print(f"Loaded existing model weights from {MODEL_PATH}.")
    return model


def fine_tune_model(
    model: CareerMatcherNN,
    x_train: torch.FloatTensor,
    y_train: torch.LongTensor,
) -> None:
    print("Starting continual fine-tuning...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:02d}/{EPOCHS} - Loss: {loss.item():.6f}")

    print("Fine-tuning complete.")


def save_model(model: CareerMatcherNN, path: Path) -> None:
    print(f"Saving fine-tuned model weights to {path}...")
    torch.save(model.state_dict(), path)
    print("Model weights saved successfully.")


def main() -> None:
    print("Career roadmap continual training pipeline started.")
    df = load_training_data(DATA_PATH)
    encoders, target_le, scaler = load_preprocessors(PREPROCESSORS_PATH)
    x_train, y_train = preprocess_dataframe(df, encoders, target_le, scaler)
    model = load_model(input_dim=x_train.shape[1], output_dim=len(target_le.classes_))
    fine_tune_model(model, x_train, y_train)
    save_model(model, MODEL_PATH)
    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
