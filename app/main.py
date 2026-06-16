from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageColor, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "app" / "static"
WORKSPACE = ROOT / "workspace"
VIDEOS = WORKSPACE / "videos"
FRAMES = WORKSPACE / "frames"
MASKS = WORKSPACE / "masks"
OUTPUTS = WORKSPACE / "outputs"
PROJECTS = WORKSPACE / "projects"
MODELS = ROOT / "models"

for folder in (VIDEOS, FRAMES, MASKS, OUTPUTS, PROJECTS):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Chrono Compositor")


class Point(BaseModel):
    x: float
    y: float
    label: int = 1


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class SegmentRequest(BaseModel):
    video_id: str
    frame_index: int = Field(ge=0)
    points: list[Point] = Field(default_factory=list)
    boxes: list[Box] = Field(default_factory=list)


class LayerRequest(BaseModel):
    video_id: str
    frame_index: int = Field(ge=0)
    mask_id: str
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    feather: int = Field(default=0, ge=0, le=80)
    outline_color: str = "#00d5ff"
    outline_width: int = Field(default=0, ge=0, le=30)


class ComposeRequest(BaseModel):
    background_video_id: str
    background_frame_index: int = Field(ge=0)
    layers: list[LayerRequest]


class CropRect(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class StoryboardFrame(BaseModel):
    video_id: str
    frame_index: int = Field(ge=0)
    crop: CropRect


class StoryboardRequest(BaseModel):
    frames: list[StoryboardFrame] = Field(min_length=1)
    target_height: int | None = Field(default=None, ge=1, le=8192)
    gap: int = Field(default=12, ge=0, le=500)
    gap_color: str = "#f4f0e8"
    format: str = "png"


class ProjectCreate(BaseModel):
    name: str = "Untitled project"


class ProjectDocument(BaseModel):
    project_id: str
    name: str
    video_id: str | None = None
    selected_frames: list[dict[str, Any]] = Field(default_factory=list)
    background_frame_index: int | None = None
    active_mode: str = "chrono"
    storyboard: dict[str, Any] = Field(default_factory=lambda: {
        "gap": 12,
        "gap_color": "#f4f0e8",
        "target_height": None,
    })
    mode_workspaces: dict[str, dict[str, Any]] = Field(default_factory=dict)
    created_at: float
    updated_at: float


@dataclass
class VideoInfo:
    video_id: str
    filename: str
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


def _meta_path(video_id: str) -> Path:
    return VIDEOS / video_id / "meta.json"


def _load_video_info(video_id: str) -> VideoInfo:
    path = _meta_path(video_id)
    if not path.exists():
        raise HTTPException(404, f"Unknown video_id: {video_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return VideoInfo(path=Path(data["path"]), **{k: v for k, v in data.items() if k != "path"})


def _write_video_info(info: VideoInfo) -> None:
    folder = VIDEOS / info.video_id
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "video_id": info.video_id,
        "filename": info.filename,
        "path": str(info.path),
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "frame_count": info.frame_count,
        "duration": info.duration,
    }
    (folder / "meta.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _safe_ext(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return ext if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"} else ".mp4"


def _project_path(project_id: str) -> Path:
    return PROJECTS / project_id / "project.json"


def _load_project(project_id: str) -> ProjectDocument:
    path = _project_path(project_id)
    if not path.exists():
        raise HTTPException(404, f"Unknown project_id: {project_id}")
    return ProjectDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _save_project(project: ProjectDocument) -> ProjectDocument:
    project.updated_at = time.time()
    path = _project_path(project.project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(project.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)
    return project


def _frame_path(video_id: str, frame_index: int) -> Path:
    folder = FRAMES / video_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"frame_{frame_index:08d}.png"


def _read_frame(video_id: str, frame_index: int) -> Path:
    info = _load_video_info(video_id)
    if frame_index < 0 or frame_index >= info.frame_count:
        raise HTTPException(400, f"frame_index out of range: 0..{info.frame_count - 1}")
    target = _frame_path(video_id, frame_index)
    if target.exists():
        return target

    ok = False
    frame = None
    for _ in range(3):
        cap = cv2.VideoCapture(str(info.path))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            break
        if target.exists():
            return target
        time.sleep(0.15)
    if not ok or frame is None:
        raise HTTPException(500, f"Cannot decode frame {frame_index}")
    tmp = target.with_name(f"{target.stem}.{uuid.uuid4().hex}.tmp.png")
    cv2.imwrite(str(tmp), frame)
    tmp.replace(target)
    return target


def _image_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


class Sam2Provider:
    def __init__(self) -> None:
        self._predictor = None
        self._error: str | None = None
        self._device: str | None = None
        self._load_lock = threading.Lock()

    @property
    def status(self) -> dict[str, Any]:
        if self._predictor is not None:
            return {"available": True, "provider": "sam2", "device": self._device, "error": None}
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
        return {
            "available": self._predictor is not None,
            "provider": "sam2",
            "device": self._device,
            "error": self._error,
        }

    def _load(self) -> None:
        with self._load_lock:
            if self._predictor is not None:
                return
            self._load_unlocked()

    def _load_unlocked(self) -> None:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        candidates = [
            ("configs/sam2.1/sam2.1_hiera_l.yaml", MODELS / "sam2.1_hiera_large.pt"),
            ("configs/sam2.1/sam2.1_hiera_b+.yaml", MODELS / "sam2.1_hiera_base_plus.pt"),
        ]
        ckpt = None
        cfg = None
        for candidate_cfg, candidate_ckpt in candidates:
            if candidate_ckpt.exists():
                cfg = candidate_cfg
                ckpt = candidate_ckpt
                break
        if ckpt is None or cfg is None:
            raise RuntimeError("SAM2 checkpoint not found. Run setup_conda.ps1 first.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam2(cfg, str(ckpt), device=device)
        self._predictor = SAM2ImagePredictor(model)
        self._device = device
        self._error = None

    def predict(self, image: np.ndarray, request: SegmentRequest) -> np.ndarray:
        self._load()
        assert self._predictor is not None
        self._predictor.set_image(image)

        point_coords = None
        point_labels = None
        box = None
        if request.points:
            point_coords = np.array([[p.x, p.y] for p in request.points], dtype=np.float32)
            point_labels = np.array([p.label for p in request.points], dtype=np.int32)
        if request.boxes:
            b = request.boxes[-1]
            box = np.array([b.x1, b.y1, b.x2, b.y2], dtype=np.float32)
        if point_coords is None and box is None:
            raise HTTPException(400, "Add at least one positive point or box.")

        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        return masks[best].astype(np.uint8) * 255


sam2_provider = Sam2Provider()


def _mask_paths(mask_id: str) -> tuple[Path, Path]:
    return MASKS / f"{mask_id}.png", MASKS / f"{mask_id}_overlay.png"


def _save_mask_preview(frame_path: Path, mask: np.ndarray, mask_id: str) -> dict[str, str]:
    mask_path, overlay_path = _mask_paths(mask_id)
    Image.fromarray(mask, mode="L").save(mask_path)

    base = Image.open(frame_path).convert("RGBA")
    color = Image.new("RGBA", base.size, (0, 213, 255, 110))
    alpha = Image.fromarray(mask, mode="L")
    overlay = Image.alpha_composite(base, Image.composite(color, Image.new("RGBA", base.size), alpha))
    overlay.save(overlay_path)
    return {
        "mask_id": mask_id,
        "mask_url": f"/api/mask/{mask_id}",
        "overlay_url": f"/api/mask/{mask_id}/overlay",
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "workspace": str(WORKSPACE),
        "sam2": sam2_provider.status,
    }


@app.get("/api/projects")
def list_projects() -> list[dict[str, Any]]:
    projects: list[dict[str, Any]] = []
    for path in PROJECTS.glob("*/project.json"):
        try:
            project = ProjectDocument.model_validate_json(path.read_text(encoding="utf-8"))
            projects.append(project.model_dump())
        except Exception:
            continue
    return sorted(projects, key=lambda item: item["updated_at"], reverse=True)


@app.post("/api/projects")
def create_project(request: ProjectCreate) -> ProjectDocument:
    now = time.time()
    project = ProjectDocument(
        project_id=uuid.uuid4().hex[:16],
        name=request.name.strip() or "Untitled project",
        created_at=now,
        updated_at=now,
    )
    return _save_project(project)


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> ProjectDocument:
    return _load_project(project_id)


@app.put("/api/projects/{project_id}")
def update_project(project_id: str, project: ProjectDocument) -> ProjectDocument:
    if project.project_id != project_id:
        raise HTTPException(400, "project_id does not match URL")
    existing = _load_project(project_id)
    project.created_at = existing.created_at
    return _save_project(project)


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str) -> dict[str, bool]:
    path = _project_path(project_id)
    if not path.exists():
        raise HTTPException(404, f"Unknown project_id: {project_id}")
    import shutil
    shutil.rmtree(path.parent)
    return {"ok": True}


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty upload")
    digest = hashlib.sha1(raw[:1024 * 1024] + str(time.time()).encode()).hexdigest()[:16]
    video_id = digest
    folder = VIDEOS / video_id
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"original{_safe_ext(file.filename or '')}"
    target.write_bytes(raw)

    cap = cv2.VideoCapture(str(target))
    if not cap.isOpened():
        raise HTTPException(400, "Cannot open uploaded video")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise HTTPException(400, "Cannot read video metadata")

    info = VideoInfo(
        video_id=video_id,
        filename=file.filename or target.name,
        path=target,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=frame_count / fps if fps else 0,
    )
    _write_video_info(info)
    return {**info.__dict__, "path": str(info.path), "stream_url": f"/api/video/{video_id}/stream"}


@app.get("/api/video/{video_id}")
def video_meta(video_id: str) -> dict[str, Any]:
    info = _load_video_info(video_id)
    return {**info.__dict__, "path": str(info.path), "stream_url": f"/api/video/{video_id}/stream"}


@app.get("/api/video/{video_id}/stream")
def stream_video(video_id: str) -> FileResponse:
    info = _load_video_info(video_id)
    return FileResponse(info.path)


@app.get("/api/video/{video_id}/frame/{frame_index}")
def frame_image(video_id: str, frame_index: int) -> FileResponse:
    return FileResponse(_read_frame(video_id, frame_index), media_type="image/png")


@app.post("/api/segment")
def segment(request: SegmentRequest) -> JSONResponse:
    frame_path = _read_frame(request.video_id, request.frame_index)
    image = _image_rgb(frame_path)
    mask = sam2_provider.predict(image, request)
    mask_id = f"{request.video_id}_{request.frame_index}_{uuid.uuid4().hex[:10]}"
    return JSONResponse(_save_mask_preview(frame_path, mask, mask_id))


@app.get("/api/mask/{mask_id}")
def get_mask(mask_id: str) -> FileResponse:
    mask_path, _ = _mask_paths(mask_id)
    if not mask_path.exists():
        raise HTTPException(404, "Unknown mask")
    return FileResponse(mask_path, media_type="image/png")


@app.get("/api/mask/{mask_id}/overlay")
def get_mask_overlay(mask_id: str) -> FileResponse:
    _, overlay_path = _mask_paths(mask_id)
    if not overlay_path.exists():
        raise HTTPException(404, "Unknown mask")
    return FileResponse(overlay_path, media_type="image/png")


def _outline_from_alpha(alpha: Image.Image, color: str, width: int) -> Image.Image:
    rgba = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
    if width <= 0:
        return rgba
    expanded = alpha.filter(ImageFilter.MaxFilter(width * 2 + 1))
    edge = Image.eval(expanded, lambda px: max(0, px))
    edge = Image.fromarray(np.maximum(np.array(edge) - np.array(alpha), 0).astype(np.uint8), mode="L")
    rgb = ImageColor.getrgb(color)
    rgba.paste((*rgb, 255), mask=edge)
    return rgba


@app.post("/api/compose")
def compose(request: ComposeRequest) -> FileResponse:
    background_path = _read_frame(request.background_video_id, request.background_frame_index)
    canvas = Image.open(background_path).convert("RGBA")

    for layer in request.layers:
        frame = Image.open(_read_frame(layer.video_id, layer.frame_index)).convert("RGBA")
        mask_path, _ = _mask_paths(layer.mask_id)
        if not mask_path.exists():
            raise HTTPException(404, f"Unknown mask: {layer.mask_id}")
        alpha = Image.open(mask_path).convert("L")
        if layer.feather > 0:
            alpha = alpha.filter(ImageFilter.GaussianBlur(layer.feather))

        if layer.opacity < 1.0:
            alpha = Image.eval(alpha, lambda px: int(px * layer.opacity))

        cutout = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        cutout.paste(frame, (0, 0), alpha)
        if layer.outline_width > 0:
            canvas = Image.alpha_composite(canvas, _outline_from_alpha(alpha, layer.outline_color, layer.outline_width))
        canvas = Image.alpha_composite(canvas, cutout)

    output_id = f"chrono_{uuid.uuid4().hex[:12]}"
    output = OUTPUTS / f"{output_id}.png"
    canvas.save(output)
    sidecar = OUTPUTS / f"{output_id}.json"
    sidecar.write_text(request.model_dump_json(indent=2), encoding="utf-8")
    return FileResponse(output, media_type="image/png", filename=f"{output_id}.png")


@app.post("/api/storyboard")
def storyboard(request: StoryboardRequest) -> FileResponse:
    crops: list[Image.Image] = []
    for item in request.frames:
        frame = Image.open(_read_frame(item.video_id, item.frame_index)).convert("RGB")
        x1 = max(0, min(frame.width - 1, int(round(item.crop.x))))
        y1 = max(0, min(frame.height - 1, int(round(item.crop.y))))
        x2 = max(x1 + 1, min(frame.width, int(round(item.crop.x + item.crop.width))))
        y2 = max(y1 + 1, min(frame.height, int(round(item.crop.y + item.crop.height))))
        crops.append(frame.crop((x1, y1, x2, y2)))

    target_height = request.target_height or crops[0].height
    resized: list[Image.Image] = []
    for crop in crops:
        width = max(1, round(crop.width * target_height / crop.height))
        resized.append(crop.resize((width, target_height), Image.Resampling.LANCZOS))

    gap_color = ImageColor.getrgb(request.gap_color)
    total_width = sum(image.width for image in resized) + request.gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total_width, target_height), gap_color)
    cursor = 0
    for image in resized:
        canvas.paste(image, (cursor, 0))
        cursor += image.width + request.gap

    extension = "jpg" if request.format.lower() in {"jpg", "jpeg"} else "png"
    output_id = f"storyboard_{uuid.uuid4().hex[:12]}"
    output = OUTPUTS / f"{output_id}.{extension}"
    canvas.save(output, quality=95)
    sidecar = OUTPUTS / f"{output_id}.json"
    sidecar.write_text(request.model_dump_json(indent=2), encoding="utf-8")
    media_type = "image/jpeg" if extension == "jpg" else "image/png"
    return FileResponse(output, media_type=media_type, filename=output.name)


@app.get("/api/exports")
def list_exports() -> list[dict[str, Any]]:
    results = []
    for path in OUTPUTS.glob("*.*"):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        results.append({
            "name": path.name,
            "url": f"/api/exports/{path.name}",
            "size": path.stat().st_size,
            "created_at": path.stat().st_mtime,
        })
    return sorted(results, key=lambda item: item["created_at"], reverse=True)


@app.get("/api/exports/{filename}")
def get_export(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    path = OUTPUTS / safe_name
    if not path.exists() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(404, "Unknown export")
    return FileResponse(path, filename=path.name)


@app.exception_handler(Exception)
async def all_exception_handler(_, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    return JSONResponse({"detail": str(exc)}, status_code=500)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
