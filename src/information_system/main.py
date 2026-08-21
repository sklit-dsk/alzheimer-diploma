"""Minimal FastAPI web layer for the research information-system prototype."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import MODEL_NAME, RESEARCH_WARNING
from .inference import InferenceInputError, predict_subject_from_tensors
from .model import InformationSystemModelError, ModelBundle, load_e07_resnet18_checkpoint
from .preprocessing import ImageValidationError, preprocess_mri_jpg_bytes


MAX_FILES_PER_REQUEST = 300
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        app.state.model_bundle = load_e07_resnet18_checkpoint()
    except InformationSystemModelError:
        raise
    except Exception as exc:
        raise RuntimeError("Failed to load the local E07 research model.") from exc
    yield


app = FastAPI(
    title="MRI Research Classification Prototype",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _render_error(
    request: Request,
    message: str,
    *,
    status_code: int = 400,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "error.html",
        {
            "message": message,
            "research_warning": RESEARCH_WARNING,
        },
        status_code=status_code,
    )


def _clean_validation_message(error: ImageValidationError) -> str:
    message = str(error)
    if "Unsupported file extension" in message:
        return "Only JPG/JPEG MRI slice images are accepted."
    if "could not be decoded" in message:
        return "One or more files could not be decoded as images."
    if "is empty" in message:
        return "One or more uploaded files are empty."
    return message


def _get_model_bundle(request: Request) -> ModelBundle:
    model_bundle = getattr(request.app.state, "model_bundle", None)
    if model_bundle is None:
        raise RuntimeError("The E07 research model is not loaded.")
    return model_bundle


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "model_name": MODEL_NAME,
            "max_files": MAX_FILES_PER_REQUEST,
            "max_file_size_mb": MAX_FILE_SIZE_BYTES // (1024 * 1024),
            "research_warning": RESEARCH_WARNING,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    files: list[UploadFile] | None = File(default=None),
) -> HTMLResponse:
    if not files or all(not file.filename for file in files):
        return _render_error(request, "No files were uploaded.")

    if len(files) > MAX_FILES_PER_REQUEST:
        return _render_error(request, "Too many files were uploaded.")

    filenames: list[str] = []
    tensors = []
    for upload in files:
        filename = upload.filename or ""
        if not filename:
            return _render_error(request, "No files were uploaded.")

        image_bytes = await upload.read()
        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            return _render_error(request, "File exceeds the maximum allowed size.")

        try:
            tensor = preprocess_mri_jpg_bytes(filename, image_bytes)
        except ImageValidationError as exc:
            return _render_error(request, _clean_validation_message(exc))

        filenames.append(filename)
        tensors.append(tensor)

    try:
        subject_prediction, _slice_predictions = predict_subject_from_tensors(
            _get_model_bundle(request),
            filenames,
            tensors,
        )
    except (InferenceInputError, RuntimeError):
        return _render_error(
            request,
            "Inference failed; check the local model setup.",
            status_code=500,
        )

    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "prediction": subject_prediction,
            "research_warning": RESEARCH_WARNING,
        },
    )
