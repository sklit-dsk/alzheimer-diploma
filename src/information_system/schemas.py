"""Small data structures returned by the inference core."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlicePrediction:
    filename: str
    probability_very_mild_dementia: float
    predicted_label_at_0_5: int
    predicted_class_at_0_5: str


@dataclass(frozen=True)
class SubjectPrediction:
    number_of_slices: int
    mean_model_score: float
    threshold: float
    predicted_label: int
    predicted_class: str
    model_name: str
    research_warning: str

