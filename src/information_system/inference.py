"""Inference and subject-level mean-probability aggregation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .config import LABEL_TO_CLASS, MODEL_NAME, RESEARCH_WARNING, THRESHOLD
from .model import ModelBundle
from .schemas import SlicePrediction, SubjectPrediction


class InferenceInputError(ValueError):
    """Raised when inference inputs are incomplete or inconsistent."""


def predict_slices_from_tensors(
    model: nn.Module,
    device: torch.device,
    filenames: Sequence[str],
    tensors: Sequence[torch.Tensor],
) -> list[SlicePrediction]:
    """Run batch inference and return one prediction per slice."""
    if not tensors:
        raise InferenceInputError("At least one preprocessed slice tensor is required.")
    if len(filenames) != len(tensors):
        raise InferenceInputError(
            f"filenames and tensors length mismatch: {len(filenames)} vs {len(tensors)}."
        )

    batch = torch.stack(list(tensors)).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()

    predictions: list[SlicePrediction] = []
    for filename, probability in zip(filenames, probabilities, strict=True):
        predicted_label = int(float(probability) >= THRESHOLD)
        predictions.append(
            SlicePrediction(
                filename=str(filename),
                probability_very_mild_dementia=float(probability),
                predicted_label_at_0_5=predicted_label,
                predicted_class_at_0_5=LABEL_TO_CLASS[predicted_label],
            )
        )
    return predictions


def aggregate_subject_mean_probability(
    slice_predictions: Sequence[SlicePrediction],
    *,
    model_name: str = MODEL_NAME,
) -> SubjectPrediction:
    """Aggregate slice predictions into one subject-level result."""
    if not slice_predictions:
        raise InferenceInputError("At least one slice prediction is required.")

    mean_score = sum(
        prediction.probability_very_mild_dementia for prediction in slice_predictions
    ) / len(slice_predictions)
    predicted_label = int(mean_score >= THRESHOLD)
    return SubjectPrediction(
        number_of_slices=len(slice_predictions),
        mean_model_score=float(mean_score),
        threshold=THRESHOLD,
        predicted_label=predicted_label,
        predicted_class=LABEL_TO_CLASS[predicted_label],
        model_name=model_name,
        research_warning=RESEARCH_WARNING,
    )


def predict_subject_from_tensors(
    model_bundle: ModelBundle,
    filenames: Sequence[str],
    tensors: Sequence[torch.Tensor],
) -> tuple[SubjectPrediction, list[SlicePrediction]]:
    """Run slice inference and return subject-level mean aggregation."""
    slice_predictions = predict_slices_from_tensors(
        model_bundle.model,
        model_bundle.device,
        filenames,
        tensors,
    )
    subject_prediction = aggregate_subject_mean_probability(
        slice_predictions,
        model_name=str(model_bundle.metadata.get("model_name", MODEL_NAME)),
    )
    return subject_prediction, slice_predictions

