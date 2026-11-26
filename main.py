import base64
import logging
import math
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

# =============================================================================
# Configuration & Ergonomic Constants
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "yolo11n-pose.pt"
INFERENCE_IMGSZ = 320  # Fast upper-body pose inference (~3-4x faster on CPU)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

# COCO Keypoint indices used for upper-body ergonomic assessment
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6

KEYPOINT_CONF_THRESHOLD = 0.40
MIN_SHOULDER_WIDTH_PX = 12.0

# Ergonomic thresholds (from PRD Section 3)
IDEAL_SHOULDER_TILT_MAX = 6.5       # degrees
DEFAULT_NECK_RATIO = 0.58           # vertical nose-to-shoulder / shoulder width
NECK_DROP_THRESHOLD = 0.18          # > 18% drop triggers Tech Neck
DEFAULT_FACE_RATIO = 0.11           # inter-eye distance / frame width
FACE_INCREASE_THRESHOLD = 0.25      # > 25% increase triggers Eye-Strain warning
SHOULDER_TO_EYE_SCALE = 0.26        # anthropometric fallback when eyes are occluded

# Visual theme colors (BGR for OpenCV) aligned with Winter Mesh Gradient palette
# Midnight Blue: #192641 -> (65, 38, 25)
# Slate:         #596779 -> (121, 103, 89)
# Powder Blue:   #A1BDD0 -> (208, 189, 161)
# Mist Blue:     #D2E0EE -> (238, 224, 210)
# Snow:          #F8F9FA -> (250, 249, 248)
BGR_MIDNIGHT = (65, 38, 25)
BGR_SLATE = (121, 103, 89)
BGR_POWDER = (208, 189, 161)
BGR_MIST = (238, 224, 210)
BGR_SNOW = (250, 249, 248)

STATUS_BGR = {
    "ideal": (129, 185, 16),    # Emerald #10B981
    "warning": (11, 158, 245),  # Amber   #F59E0B
    "poor": (94, 63, 244),      # Rose    #F43F5E
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ergovision")

pose_model: Optional[YOLO] = None
model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load YOLO11n-pose once at server startup and warm it up in memory."""
    global pose_model
    logger.info("Memuat model pose %s...", MODEL_NAME)
    pose_model = YOLO(MODEL_NAME)
    # Warm-up pass in RAM to avoid first-request latency spike
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    with model_lock:
        pose_model.predict(dummy_frame, imgsz=INFERENCE_IMGSZ, verbose=False, conf=0.25)
    logger.info("Model %s siap menerima inferensi real-time (imgsz=%d).", MODEL_NAME, INFERENCE_IMGSZ)
    yield
    logger.info("Menutup server ErgoVision AI.")


app = FastAPI(
    title="ErgoVision AI",
    description="Real-time AI Posture & Eye-Strain Monitor berbasis YOLO11n-pose (Zero-Storage Privacy)",
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# =============================================================================
# Helper Functions: Geometry, Scoring & Minimalist Skeleton Rendering
# =============================================================================
def _encode_jpeg_data_uri(frame: np.ndarray, quality: int = 80) -> str:
    """Encode BGR frame in RAM to a base64 JPEG data URI."""
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Gagal mengodekan frame output ke JPEG.")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _sanitize_baseline(value: Optional[float], low: float, high: float, default: float) -> float:
    """Validate optional user-calibrated baseline values."""
    if value is None or math.isnan(value) or math.isinf(value):
        return default
    if low <= value <= high:
        return float(value)
    return default


def _draw_dashed_horizontal(
    canvas: np.ndarray,
    center_x: int,
    y: int,
    half_span: int,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw a subtle horizontal reference guide across shoulder midpoint."""
    start_x = max(0, center_x - half_span)
    end_x = min(canvas.shape[1] - 1, center_x + half_span)
    dash_len = 8
    gap_len = 6
    x = start_x
    while x < end_x:
        x2 = min(x + dash_len, end_x)
        cv2.line(canvas, (x, y), (x2, y), color, thickness, cv2.LINE_AA)
        x += dash_len + gap_len


def _draw_joint(
    canvas: np.ndarray,
    point: Tuple[int, int],
    fill_color: Tuple[int, int, int],
    radius: int = 6,
) -> None:
    """Draw a clean anti-aliased joint marker with Snow outer ring."""
    cv2.circle(canvas, point, radius + 3, BGR_MIDNIGHT, -1, cv2.LINE_AA)
    cv2.circle(canvas, point, radius + 1, BGR_SNOW, -1, cv2.LINE_AA)
    cv2.circle(canvas, point, max(2, radius - 1), fill_color, -1, cv2.LINE_AA)


def _draw_minimalist_skeleton(
    frame: np.ndarray,
    nose: Tuple[float, float],
    left_shoulder: Tuple[float, float],
    right_shoulder: Tuple[float, float],
    mid_shoulder: Tuple[float, float],
    left_eye: Optional[Tuple[float, float]],
    right_eye: Optional[Tuple[float, float]],
    status: str,
    shoulder_tilted: bool,
    tech_neck: bool,
    too_close: bool,
) -> np.ndarray:
    """Render minimalist, high-contrast ergonomic guide lines in RAM."""
    annotated = frame.copy()
    overlay = frame.copy()

    pt_nose = (int(round(nose[0])), int(round(nose[1])))
    pt_ls = (int(round(left_shoulder[0])), int(round(left_shoulder[1])))
    pt_rs = (int(round(right_shoulder[0])), int(round(right_shoulder[1])))
    pt_mid = (int(round(mid_shoulder[0])), int(round(mid_shoulder[1])))

    status_color = STATUS_BGR.get(status, BGR_POWDER)
    shoulder_color = STATUS_BGR["warning"] if shoulder_tilted else BGR_POWDER
    neck_color = STATUS_BGR["poor"] if tech_neck else BGR_MIST
    eye_color = STATUS_BGR["poor"] if too_close else BGR_POWDER

    # Subtle horizontal level reference across shoulder midpoint
    half_ref = max(40, int(abs(pt_ls[0] - pt_rs[0]) * 0.58))
    _draw_dashed_horizontal(overlay, pt_mid[0], pt_mid[1], half_ref, BGR_SNOW, thickness=1)

    # Soft backing strokes for visibility on bright or dark backgrounds
    cv2.line(overlay, pt_ls, pt_rs, BGR_MIDNIGHT, 6, cv2.LINE_AA)
    cv2.line(overlay, pt_mid, pt_nose, BGR_MIDNIGHT, 5, cv2.LINE_AA)

    # Blend subtle backing
    cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

    # Crisp primary ergonomic axes
    cv2.line(annotated, pt_ls, pt_rs, shoulder_color, 3, cv2.LINE_AA)
    cv2.line(annotated, pt_mid, pt_nose, neck_color, 2, cv2.LINE_AA)

    # Optional inter-eye proximity line
    if left_eye is not None and right_eye is not None:
        pt_le = (int(round(left_eye[0])), int(round(left_eye[1])))
        pt_re = (int(round(right_eye[0])), int(round(right_eye[1])))
        cv2.line(annotated, pt_le, pt_re, eye_color, 2, cv2.LINE_AA)
        _draw_joint(annotated, pt_le, eye_color, radius=4)
        _draw_joint(annotated, pt_re, eye_color, radius=4)

    # Key ergonomic nodes
    _draw_joint(annotated, pt_ls, shoulder_color, radius=6)
    _draw_joint(annotated, pt_rs, shoulder_color, radius=6)
    _draw_joint(annotated, pt_mid, status_color, radius=5)
    _draw_joint(annotated, pt_nose, status_color, radius=6)

    return annotated


def _analyze_frame_sync(
    frame: np.ndarray,
    baseline_neck_ratio: Optional[float],
    baseline_face_ratio: Optional[float],
    include_image: bool = True,
) -> Dict[str, Any]:
    """Run YOLO11n-pose inference, ergonomic calculations, scoring, and optional drawing."""
    if pose_model is None:
        raise RuntimeError("Model YOLO belum diinisialisasi.")

    frame_height, frame_width = frame.shape[:2]

    with model_lock:
        results = pose_model.predict(
            frame,
            imgsz=INFERENCE_IMGSZ,
            verbose=False,
            conf=0.25,
            max_det=3,
        )

    if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return _undetected_response(frame, include_image=include_image)

    kps_xy = results[0].keypoints.xy[0].cpu().numpy()
    kps_conf = (
        results[0].keypoints.conf[0].cpu().numpy()
        if results[0].keypoints.conf is not None
        else np.ones(len(kps_xy), dtype=np.float32)
    )

    if len(kps_xy) <= KP_RIGHT_SHOULDER:
        return _undetected_response(frame, include_image=include_image)

    nose_conf = float(kps_conf[KP_NOSE])
    ls_conf = float(kps_conf[KP_LEFT_SHOULDER])
    rs_conf = float(kps_conf[KP_RIGHT_SHOULDER])

    # PRD Section 3A: Confidence validation for nose & both shoulders
    if (
        nose_conf < KEYPOINT_CONF_THRESHOLD
        or ls_conf < KEYPOINT_CONF_THRESHOLD
        or rs_conf < KEYPOINT_CONF_THRESHOLD
    ):
        return _undetected_response(frame, include_image=include_image)

    nose = (float(kps_xy[KP_NOSE][0]), float(kps_xy[KP_NOSE][1]))
    left_shoulder = (float(kps_xy[KP_LEFT_SHOULDER][0]), float(kps_xy[KP_LEFT_SHOULDER][1]))
    right_shoulder = (float(kps_xy[KP_RIGHT_SHOULDER][0]), float(kps_xy[KP_RIGHT_SHOULDER][1]))

    dx_shoulder = left_shoulder[0] - right_shoulder[0]
    dy_shoulder = left_shoulder[1] - right_shoulder[1]
    shoulder_width = math.hypot(dx_shoulder, dy_shoulder)

    if shoulder_width < MIN_SHOULDER_WIDTH_PX:
        return _undetected_response(frame, include_image=include_image)

    # -------------------------------------------------------------------------
    # 1. Shoulder Tilt (degrees relative to horizontal axis)
    # -------------------------------------------------------------------------
    shoulder_tilt = abs(
        math.degrees(math.atan2(abs(dy_shoulder), max(abs(dx_shoulder), 1e-6)))
    )
    shoulder_tilted = shoulder_tilt > IDEAL_SHOULDER_TILT_MAX

    # -------------------------------------------------------------------------
    # 2. Tech Neck / Slouching (Neck Ratio)
    # -------------------------------------------------------------------------
    mid_shoulder = (
        (left_shoulder[0] + right_shoulder[0]) / 2.0,
        (left_shoulder[1] + right_shoulder[1]) / 2.0,
    )
    vertical_neck = max(mid_shoulder[1] - nose[1], 0.0)
    neck_ratio = vertical_neck / shoulder_width

    target_neck = _sanitize_baseline(
        baseline_neck_ratio, low=0.15, high=1.60, default=DEFAULT_NECK_RATIO
    )
    neck_drop = max(0.0, (target_neck - neck_ratio) / max(target_neck, 1e-6))
    tech_neck = neck_drop > NECK_DROP_THRESHOLD

    if tech_neck:
        neck_status = "Menunduk"
    elif neck_drop > 0.10:
        neck_status = "Agak Menunduk"
    else:
        neck_status = "Tegak"

    # -------------------------------------------------------------------------
    # 3. Eye-Strain Proximity (Face Ratio)
    # -------------------------------------------------------------------------
    le_conf = float(kps_conf[KP_LEFT_EYE])
    re_conf = float(kps_conf[KP_RIGHT_EYE])
    eyes_detected = le_conf >= KEYPOINT_CONF_THRESHOLD and re_conf >= KEYPOINT_CONF_THRESHOLD

    left_eye: Optional[Tuple[float, float]] = None
    right_eye: Optional[Tuple[float, float]] = None

    if eyes_detected:
        left_eye = (float(kps_xy[KP_LEFT_EYE][0]), float(kps_xy[KP_LEFT_EYE][1]))
        right_eye = (float(kps_xy[KP_RIGHT_EYE][0]), float(kps_xy[KP_RIGHT_EYE][1]))
        eye_dist = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
        face_ratio = eye_dist / max(float(frame_width), 1.0)
    else:
        # Fallback to shoulder width scaled to inter-eye anthropometric scale
        face_ratio = (shoulder_width * SHOULDER_TO_EYE_SCALE) / max(float(frame_width), 1.0)

    target_face = _sanitize_baseline(
        baseline_face_ratio, low=0.03, high=0.50, default=DEFAULT_FACE_RATIO
    )
    face_increase = max(0.0, (face_ratio - target_face) / max(target_face, 1e-6))
    too_close = face_increase > FACE_INCREASE_THRESHOLD

    if too_close:
        distance_status = "Terlalu Dekat"
    elif face_increase > 0.12:
        distance_status = "Agak Dekat"
    else:
        distance_status = "Aman"

    # -------------------------------------------------------------------------
    # 4. Ergonomic Scoring Engine (0 - 100)
    # -------------------------------------------------------------------------
    if shoulder_tilt <= IDEAL_SHOULDER_TILT_MAX:
        tilt_penalty = max(0.0, (shoulder_tilt - 4.5) * 1.5)
    else:
        tilt_penalty = 15.0 + min(25.0, (shoulder_tilt - IDEAL_SHOULDER_TILT_MAX) * 2.5)

    if neck_drop <= NECK_DROP_THRESHOLD:
        neck_penalty = max(0.0, (neck_drop - 0.08) * 50.0)
    else:
        neck_penalty = 21.0 + min(29.0, (neck_drop - NECK_DROP_THRESHOLD) * 110.0)

    if face_increase <= FACE_INCREASE_THRESHOLD:
        face_penalty = max(0.0, (face_increase - 0.12) * 35.0)
    else:
        face_penalty = 18.0 + min(27.0, (face_increase - FACE_INCREASE_THRESHOLD) * 75.0)

    raw_score = 100.0 - (tilt_penalty + neck_penalty + face_penalty)
    score = int(round(max(0.0, min(100.0, raw_score))))

    if score >= 80:
        status = "ideal"
        status_label = "Postur Bagus & Sehat"
    elif score >= 55:
        status = "warning"
        status_label = "Perlu Diperbaiki Sedikit"
    else:
        status = "poor"
        status_label = "Postur Kurang Baik"

    # -------------------------------------------------------------------------
    # 5. Issues & Actionable Recommendations
    # -------------------------------------------------------------------------
    issues: List[Dict[str, str]] = []

    if tech_neck:
        drop_pct = round(neck_drop * 100)
        issues.append(
            {
                "type": "tech_neck",
                "severity": "high" if neck_drop > 0.30 else "medium",
                "title": "Kepala & Leher Terlalu Menunduk",
                "detail": f"Posisi kepala turun {drop_pct}% dari posisi duduk tegak Anda.",
                "recommendation": "Tegakkan punggung, tarik dagu sedikit ke belakang, dan sejajarkan layar dengan pandangan mata.",
            }
        )

    if too_close:
        inc_pct = round(face_increase * 100)
        issues.append(
            {
                "type": "eye_strain",
                "severity": "high" if face_increase > 0.42 else "medium",
                "title": "Wajah Terlalu Dekat ke Layar",
                "detail": f"Wajah Anda {inc_pct}% lebih dekat dari jarak aman ke layar.",
                "recommendation": "Mundurkan posisi duduk sekitar 50–70 cm (sepanjang lengan) agar mata tidak cepat lelah.",
            }
        )

    if shoulder_tilted:
        issues.append(
            {
                "type": "shoulder_tilt",
                "severity": "high" if shoulder_tilt > 12.0 else "medium",
                "title": "Posisi Bahu Miring",
                "detail": f"Bahu Anda miring {shoulder_tilt:.1f}° (batas ideal maksimal {IDEAL_SHOULDER_TILT_MAX}°).",
                "recommendation": "Sejajarkan kedua bahu dengan santai dan hindari bersandar hanya di satu sisi.",
            }
        )

    # Normalized [0..1] keypoints for buttery-smooth 60 FPS client-side canvas overlay
    inv_w = 1.0 / max(float(frame_width), 1.0)
    inv_h = 1.0 / max(float(frame_height), 1.0)
    normalized_keypoints: Dict[str, Any] = {
        "nose": [round(nose[0] * inv_w, 4), round(nose[1] * inv_h, 4)],
        "left_shoulder": [round(left_shoulder[0] * inv_w, 4), round(left_shoulder[1] * inv_h, 4)],
        "right_shoulder": [round(right_shoulder[0] * inv_w, 4), round(right_shoulder[1] * inv_h, 4)],
        "mid_shoulder": [round(mid_shoulder[0] * inv_w, 4), round(mid_shoulder[1] * inv_h, 4)],
        "left_eye": [round(left_eye[0] * inv_w, 4), round(left_eye[1] * inv_h, 4)] if left_eye else None,
        "right_eye": [round(right_eye[0] * inv_w, 4), round(right_eye[1] * inv_h, 4)] if right_eye else None,
        "flags": {
            "shoulder_tilted": shoulder_tilted,
            "tech_neck": tech_neck,
            "too_close": too_close,
        },
    }

    image_data_uri: Optional[str] = None
    if include_image:
        annotated = _draw_minimalist_skeleton(
            frame=frame,
            nose=nose,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
            mid_shoulder=mid_shoulder,
            left_eye=left_eye,
            right_eye=right_eye,
            status=status,
            shoulder_tilted=shoulder_tilted,
            tech_neck=tech_neck,
            too_close=too_close,
        )
        image_data_uri = _encode_jpeg_data_uri(annotated, quality=80)

    return {
        "detected": True,
        "image": image_data_uri,
        "keypoints": normalized_keypoints,
        "score": score,
        "status": status,
        "status_label": status_label,
        "metrics": {
            "shoulder_tilt": round(shoulder_tilt, 1),
            "neck_ratio": round(neck_ratio, 3),
            "face_ratio": round(face_ratio, 3),
            "neck_status": neck_status,
            "distance_status": distance_status,
            "shoulder_status": "Miring" if shoulder_tilted else "Sejajar",
            "neck_drop_pct": round(neck_drop * 100, 1),
            "face_increase_pct": round(face_increase * 100, 1),
            "target_neck": round(target_neck, 3),
            "target_face": round(target_face, 3),
        },
        "issues": issues,
    }


def _undetected_response(frame: np.ndarray, include_image: bool = True) -> Dict[str, Any]:
    """Return structured undetected payload when key joints are missing or low confidence."""
    return {
        "detected": False,
        "image": _encode_jpeg_data_uri(frame, quality=75) if include_image else None,
        "keypoints": None,
        "score": None,
        "status": "undetected",
        "status_label": "Wajah/Bahu Belum Terlihat",
        "message": "Arahkan wajah dan kedua bahu Anda ke depan kamera di tempat yang cukup terang.",
        "metrics": None,
        "issues": [
            {
                "type": "detection",
                "severity": "info",
                "title": "Arahkan Wajah & Bahu ke Kamera",
                "detail": "Kamera belum dapat melihat wajah dan kedua bahu Anda secara jelas.",
                "recommendation": "Duduk menghadap kamera dan pastikan kepala hingga kedua bahu terlihat penuh.",
            }
        ],
    }


# =============================================================================
# API Routes
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render main ErgoVision AI web dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "default_neck_ratio": DEFAULT_NECK_RATIO,
            "default_face_ratio": DEFAULT_FACE_RATIO,
            "ideal_shoulder_tilt": IDEAL_SHOULDER_TILT_MAX,
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestrators (HF Spaces / Railway)."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": pose_model is not None,
    }


@app.post("/api/analyze")
async def analyze_posture(
    file: UploadFile = File(...),
    baseline_neck_ratio: Optional[float] = Form(default=None),
    baseline_face_ratio: Optional[float] = Form(default=None),
    include_image: bool = Form(default=True),
):
    """Analyze a single webcam frame purely in RAM and return ergonomic telemetry."""
    content_type = (file.content_type or "").lower().strip()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Format gambar tidak didukung. Hanya JPEG, PNG, atau WEBP yang diperbolehkan.",
        )

    # Read at most MAX_UPLOAD_BYTES + 1 to enforce 5 MB cap without exhausting memory
    raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="File gambar kosong.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Ukuran file melebihi batas maksimal 5 MB.",
        )

    # Zero-Storage Privacy: Decode strictly in RAM
    np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Gambar tidak valid atau gagal didekode.",
        )

    result = await run_in_threadpool(
        _analyze_frame_sync,
        frame,
        baseline_neck_ratio,
        baseline_face_ratio,
        include_image,
    )
    return JSONResponse(content=result)
