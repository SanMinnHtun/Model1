from __future__ import annotations

import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch import nn


MODEL_PATH = Path("enhanced_career_matcher_nn.pth")
PREPROCESSORS_PATH = Path("career_matcher_preprocessors.pkl")

FEATURE_ORDER = [
    "programming_base",
    "study_duration",
    "preferred_task",
    "academic_strength",
    "personality_style",
]
CATEGORICAL_FEATURES = [
    "programming_base",
    "preferred_task",
    "academic_strength",
    "personality_style",
]


class CareerMatcherNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CareerMatcherNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


class StudentAnswersRequest(BaseModel):
    programming_base: str
    study_duration: int
    preferred_task: str
    academic_strength: str
    personality_style: str


class ModelState:
    model: CareerMatcherNN | None = None
    encoders: dict[str, Any] | None = None
    target_le: Any = None
    scaler: Any = None


state = ModelState()


def _get_artifact(artifacts: dict[str, Any], primary_key: str, *fallback_keys: str) -> Any:
    for key in (primary_key, *fallback_keys):
        if key in artifacts:
            return artifacts[key]
    expected_keys = ", ".join((primary_key, *fallback_keys))
    raise KeyError(f"Missing required artifact key. Expected one of: {expected_keys}")


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Accept weights saved from either this class or an equivalent Sequential MLP."""
    sequential_to_named = {
        "network.0.weight": "fc1.weight",
        "network.0.bias": "fc1.bias",
        "network.2.weight": "fc2.weight",
        "network.2.bias": "fc2.bias",
        "network.4.weight": "fc3.weight",
        "network.4.bias": "fc3.bias",
    }
    return {sequential_to_named.get(key, key): value for key, value in state_dict.items()}


def load_model_and_preprocessors() -> None:
    if not PREPROCESSORS_PATH.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {PREPROCESSORS_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights file not found: {MODEL_PATH}")

    with PREPROCESSORS_PATH.open("rb") as file:
        artifacts = pickle.load(file)

    state.encoders = _get_artifact(artifacts, "encoders", "feature_encoders")
    state.target_le = _get_artifact(artifacts, "target_le", "target_encoder")
    state.scaler = _get_artifact(artifacts, "scaler", "duration_scaler")

    output_dim = len(state.target_le.classes_)
    model = CareerMatcherNN(input_dim=5, output_dim=output_dim)
    raw_state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(_normalize_state_dict_keys(raw_state_dict))
    model.eval()

    state.model = model


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_preprocessors()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _encode_request(request: StudentAnswersRequest) -> torch.FloatTensor:
    if state.encoders is None or state.scaler is None:
        raise HTTPException(status_code=500, detail="Preprocessors are not loaded.")

    raw_values = request.dict()
    encoded_values: dict[str, float] = {}

    try:
        for feature in CATEGORICAL_FEATURES:
            encoder = state.encoders[feature]
            value = raw_values[feature]
            encoded_values[feature] = float(encoder.transform([value])[0])

        duration = pd.DataFrame({"study_duration": [float(request.study_duration)]})
        encoded_values["study_duration"] = float(state.scaler.transform(duration)[0][0])

        features = [[encoded_values[feature] for feature in FEATURE_ORDER]]
        return torch.FloatTensor(features)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not transform input values: {exc}",
        ) from exc


@app.post("/api/v1/predict-roadmap")
def predict_roadmap(request: StudentAnswersRequest) -> dict[str, Any]:
    if state.model is None or state.target_le is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    input_tensor = _encode_request(request)

    try:
        with torch.no_grad():
            logits = state.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        top_index = int(torch.argmax(probabilities).item())
        recommended_roadmap = str(state.target_le.inverse_transform([top_index])[0])
        chart_data = {
            str(class_name): round(float(probability.item()) * 100, 2)
            for class_name, probability in zip(state.target_le.classes_, probabilities)
        }

        return {
            "recommended_roadmap": recommended_roadmap,
            "chart_data": chart_data,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
