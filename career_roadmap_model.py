"""
Production-ready PyTorch MLP for recommending tailored IT career roadmaps.

This script:
1. Generates a balanced synthetic dataset with realistic rules and 5% label noise.
2. Encodes categorical features with LabelEncoder and scales study_duration.
3. Trains a PyTorch MLP classifier.
4. Evaluates test accuracy.
5. Exports model weights and preprocessing artifacts.
6. Provides an interactive prediction function for raw student inputs.
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RANDOM_SEED = 42


@dataclass(frozen=True)
class CareerMatcherConfig:
    dataset_size: int = 1_800
    noise_rate: float = 0.05
    test_size: float = 0.20
    batch_size: int = 64
    epochs: int = 120
    learning_rate: float = 0.005
    model_path: Path = Path("enhanced_career_matcher_nn.pth")
    artifacts_path: Path = Path("career_matcher_preprocessors.pkl")


class CareerRoadmapSchema:
    programming_base = ["Java", "Python", "JavaScript", "C++", "None"]
    preferred_task = [
        "Logic_Writing",
        "Data_Analysis",
        "Automation",
        "UI_Design",
        "System_Security",
    ]
    academic_strength = [
        "Problem_Solving",
        "Math_and_Logic",
        "Organized",
        "Creative_Arts",
        "Networking",
    ]
    personality_style = ["Analytical", "Creative", "Organized", "Investigative"]
    roadmaps = [
        "Backend_Roadmap",
        "Data_Science_Roadmap",
        "DevOps_Roadmap",
        "Frontend_Roadmap",
        "UI_UX_Roadmap",
        "Cyber_Security_Roadmap",
    ]
    categorical_features = [
        "programming_base",
        "preferred_task",
        "academic_strength",
        "personality_style",
    ]
    numeric_features = ["study_duration"]
    feature_order = [
        "programming_base",
        "study_duration",
        "preferred_task",
        "academic_strength",
        "personality_style",
    ]
    target_column = "roadmap"


class SyntheticCareerDataGenerator:
    """Creates balanced synthetic training data from realistic career-matching rules."""

    def __init__(self, config: CareerMatcherConfig, schema: CareerRoadmapSchema) -> None:
        self.config = config
        self.schema = schema

    def generate(self) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        rows_per_class = self.config.dataset_size // len(self.schema.roadmaps)
        noisy_rows_per_class = int(round(rows_per_class * self.config.noise_rate))

        for roadmap_index, roadmap in enumerate(self.schema.roadmaps):
            noise_indices = set(random.sample(range(rows_per_class), noisy_rows_per_class))
            noise_target = self.schema.roadmaps[
                (roadmap_index + 1) % len(self.schema.roadmaps)
            ]

            for row_index in range(rows_per_class):
                row = self._sample_profile_for_roadmap(roadmap)
                row[self.schema.target_column] = (
                    noise_target if row_index in noise_indices else roadmap
                )
                rows.append(row)

        remaining_rows = self.config.dataset_size - len(rows)
        for _ in range(remaining_rows):
            roadmap = random.choice(self.schema.roadmaps)
            row = self._sample_profile_for_roadmap(roadmap)
            row[self.schema.target_column] = roadmap
            rows.append(row)

        random.shuffle(rows)
        return pd.DataFrame(rows, columns=self.schema.feature_order + [self.schema.target_column])

    def _sample_profile_for_roadmap(self, roadmap: str) -> Dict[str, object]:
        profile_rules = {
            "Backend_Roadmap": {
                "programming_base": ["Java", "Python", "C++"],
                "study_duration": [3, 6, 9, 12, 18],
                "preferred_task": ["Logic_Writing", "Automation"],
                "academic_strength": ["Problem_Solving", "Math_and_Logic", "Organized"],
                "personality_style": ["Analytical", "Organized", "Investigative"],
            },
            "Data_Science_Roadmap": {
                "programming_base": ["Python", "None", "C++"],
                "study_duration": [3, 6, 9, 12, 18, 24],
                "preferred_task": ["Data_Analysis", "Automation"],
                "academic_strength": ["Math_and_Logic", "Problem_Solving"],
                "personality_style": ["Analytical", "Investigative"],
            },
            "DevOps_Roadmap": {
                "programming_base": ["Python", "JavaScript", "Java", "None"],
                "study_duration": [3, 6, 9, 12, 18],
                "preferred_task": ["Automation", "System_Security"],
                "academic_strength": ["Organized", "Networking", "Problem_Solving"],
                "personality_style": ["Organized", "Analytical"],
            },
            "Frontend_Roadmap": {
                "programming_base": ["JavaScript", "None", "Python"],
                "study_duration": [0, 3, 6, 9, 12],
                "preferred_task": ["UI_Design", "Logic_Writing"],
                "academic_strength": ["Creative_Arts", "Problem_Solving", "Organized"],
                "personality_style": ["Creative", "Organized", "Analytical"],
            },
            "UI_UX_Roadmap": {
                "programming_base": ["None", "JavaScript", "Python"],
                "study_duration": [0, 3, 6, 9],
                "preferred_task": ["UI_Design", "Data_Analysis"],
                "academic_strength": ["Creative_Arts", "Organized"],
                "personality_style": ["Creative", "Investigative", "Organized"],
            },
            "Cyber_Security_Roadmap": {
                "programming_base": ["C++", "Python", "Java", "None"],
                "study_duration": [3, 6, 9, 12, 18, 24],
                "preferred_task": ["System_Security", "Logic_Writing"],
                "academic_strength": ["Networking", "Problem_Solving", "Math_and_Logic"],
                "personality_style": ["Investigative", "Analytical", "Organized"],
            },
        }

        rules = profile_rules[roadmap]
        return {
            "programming_base": random.choice(rules["programming_base"]),
            "study_duration": random.choice(rules["study_duration"]),
            "preferred_task": random.choice(rules["preferred_task"]),
            "academic_strength": random.choice(rules["academic_strength"]),
            "personality_style": random.choice(rules["personality_style"]),
        }


class CareerDataPreprocessor:
    """Encodes categorical values, scales duration, and transforms targets."""

    def __init__(self, schema: CareerRoadmapSchema) -> None:
        self.schema = schema
        self.feature_encoders: Dict[str, LabelEncoder] = {}
        self.target_encoder = LabelEncoder()
        self.duration_scaler = StandardScaler()

    def fit_transform(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        transformed = data.copy()

        for feature in self.schema.categorical_features:
            encoder = LabelEncoder()
            transformed[feature] = encoder.fit_transform(transformed[feature])
            self.feature_encoders[feature] = encoder

        transformed[self.schema.numeric_features] = self.duration_scaler.fit_transform(
            transformed[self.schema.numeric_features]
        )
        y = self.target_encoder.fit_transform(transformed[self.schema.target_column])
        x = transformed[self.schema.feature_order].to_numpy(dtype=np.float32)
        return x, y.astype(np.int64)

    def transform_single(self, student_profile: Dict[str, object]) -> np.ndarray:
        self._validate_profile(student_profile)

        encoded_values: Dict[str, float] = {}
        for feature in self.schema.categorical_features:
            value = str(student_profile[feature])
            encoder = self.feature_encoders[feature]
            if value not in encoder.classes_:
                valid_values = ", ".join(encoder.classes_)
                raise ValueError(f"Unknown {feature}={value!r}. Valid values: {valid_values}")
            encoded_values[feature] = float(encoder.transform([value])[0])

        duration = pd.DataFrame(
            {"study_duration": [float(student_profile["study_duration"])]}
        )
        encoded_values["study_duration"] = float(
            self.duration_scaler.transform(duration)[0][0]
        )

        ordered_features = [encoded_values[feature] for feature in self.schema.feature_order]
        return np.array([ordered_features], dtype=np.float32)

    def decode_prediction(self, class_index: int) -> str:
        return str(self.target_encoder.inverse_transform([class_index])[0])

    def save(self, path: Path) -> None:
        artifacts = {
            "feature_encoders": self.feature_encoders,
            "target_encoder": self.target_encoder,
            "duration_scaler": self.duration_scaler,
            "feature_order": self.schema.feature_order,
            "categorical_features": self.schema.categorical_features,
            "numeric_features": self.schema.numeric_features,
        }
        with path.open("wb") as file:
            pickle.dump(artifacts, file)

    @staticmethod
    def load(path: Path, schema: CareerRoadmapSchema) -> "CareerDataPreprocessor":
        with path.open("rb") as file:
            artifacts = pickle.load(file)

        preprocessor = CareerDataPreprocessor(schema)
        preprocessor.feature_encoders = artifacts["feature_encoders"]
        preprocessor.target_encoder = artifacts["target_encoder"]
        preprocessor.duration_scaler = artifacts["duration_scaler"]
        return preprocessor

    def _validate_profile(self, student_profile: Dict[str, object]) -> None:
        missing_features = [
            feature for feature in self.schema.feature_order if feature not in student_profile
        ]
        if missing_features:
            raise ValueError(f"Missing required feature(s): {', '.join(missing_features)}")


class CareerRoadmapMLP(nn.Module):
    """MLP with input dim 5, hidden layers 64 and 32, and roadmap class logits."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class CareerMatcherNN(CareerRoadmapMLP):
    """Compatibility name used by production inference and retraining scripts."""


class CareerRoadmapTrainer:
    def __init__(
        self,
        model: CareerRoadmapMLP,
        config: CareerMatcherConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        features = torch.FloatTensor(x_train)
        targets = torch.LongTensor(y_train)
        dataset = TensorDataset(features, targets)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            total_loss = 0.0

            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(batch_features)
                loss = self.loss_fn(logits, batch_targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch_features.size(0)

            if epoch % 20 == 0 or epoch == 1:
                average_loss = total_loss / len(dataset)
                print(f"Epoch [{epoch:03d}/{self.config.epochs}] - Loss: {average_loss:.4f}")

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray, class_names: np.ndarray) -> float:
        self.model.eval()
        test_features = torch.FloatTensor(x_test).to(self.device)

        with torch.no_grad():
            logits = self.model(test_features)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

        accuracy = accuracy_score(y_test, predictions)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=class_names))
        return float(accuracy)

    def save_model(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)


class CareerRoadmapPredictor:
    def __init__(
        self,
        model: CareerRoadmapMLP,
        preprocessor: CareerDataPreprocessor,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.preprocessor = preprocessor
        self.device = device

    def predict(self, student_profile: Dict[str, object]) -> str:
        self.model.eval()
        features = self.preprocessor.transform_single(student_profile)
        tensor_features = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_features)
            predicted_class = int(torch.argmax(logits, dim=1).item())

        return self.preprocessor.decode_prediction(predicted_class)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_and_train_model() -> tuple[CareerRoadmapMLP, CareerDataPreprocessor, torch.device]:
    set_reproducibility(RANDOM_SEED)

    config = CareerMatcherConfig()
    schema = CareerRoadmapSchema()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_generator = SyntheticCareerDataGenerator(config, schema)
    dataset = data_generator.generate()
    print(f"Generated dataset shape: {dataset.shape}")
    print("Class distribution:")
    print(dataset[schema.target_column].value_counts().sort_index())

    preprocessor = CareerDataPreprocessor(schema)
    x, y = preprocessor.fit_transform(dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model = CareerRoadmapMLP(
        input_dim=len(schema.feature_order),
        output_dim=len(preprocessor.target_encoder.classes_),
    )
    trainer = CareerRoadmapTrainer(model, config, device)
    trainer.train(x_train, y_train)
    trainer.evaluate(x_test, y_test, preprocessor.target_encoder.classes_)

    trainer.save_model(config.model_path)
    preprocessor.save(config.artifacts_path)
    print(f"\nSaved model weights to: {config.model_path}")
    print(f"Saved preprocessing artifacts to: {config.artifacts_path}")

    return model, preprocessor, device


def predict_student_roadmap(
    model: CareerRoadmapMLP,
    preprocessor: CareerDataPreprocessor,
    device: torch.device,
    programming_base: str,
    study_duration: float,
    preferred_task: str,
    academic_strength: str,
    personality_style: str,
) -> str:
    """Predict and print the decoded roadmap for one raw student profile."""

    student_profile = {
        "programming_base": programming_base,
        "study_duration": study_duration,
        "preferred_task": preferred_task,
        "academic_strength": academic_strength,
        "personality_style": personality_style,
    }
    predictor = CareerRoadmapPredictor(model, preprocessor, device)
    prediction = predictor.predict(student_profile)
    print(f"Recommended Career Roadmap: {prediction}")
    return prediction


def load_exported_predictor(
    model_path: Path = Path("enhanced_career_matcher_nn.pth"),
    artifacts_path: Path = Path("career_matcher_preprocessors.pkl"),
) -> CareerRoadmapPredictor:
    """Load saved weights and preprocessing artifacts for inference-only use."""

    schema = CareerRoadmapSchema()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = CareerDataPreprocessor.load(artifacts_path, schema)
    model = CareerRoadmapMLP(
        input_dim=len(schema.feature_order),
        output_dim=len(preprocessor.target_encoder.classes_),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return CareerRoadmapPredictor(model, preprocessor, device)


if __name__ == "__main__":
    trained_model, fitted_preprocessor, runtime_device = build_and_train_model()

    # Example playground call. Edit these values or call predict_student_roadmap()
    # from another Python file/notebook with raw student inputs.
    predict_student_roadmap(
        model=trained_model,
        preprocessor=fitted_preprocessor,
        device=runtime_device,
        programming_base="Python",
        study_duration=6,
        preferred_task="Data_Analysis",
        academic_strength="Math_and_Logic",
        personality_style="Analytical",
    )
