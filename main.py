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

# COCO Keypoint indices used for upper-body ergonomic assessment (front & side view)
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12

KEYPOINT_CONF_THRESHOLD = 0.40
MIN_SHOULDER_WIDTH_PX = 12.0

# Ergonomic thresholds (front view)
IDEAL_SHOULDER_TILT_MAX = 6.5       # degrees
DEFAULT_NECK_RATIO = 0.58           # vertical nose-to-shoulder / shoulder width
NECK_DROP_THRESHOLD = 0.18          # > 18% drop triggers Tech Neck
DEFAULT_FACE_RATIO = 0.11           # inter-eye distance / frame width
FACE_INCREASE_THRESHOLD = 0.25      # > 25% increase triggers Eye-Strain warning
SHOULDER_TO_EYE_SCALE = 0.26        # anthropometric fallback when eyes are occluded

# Ergonomic thresholds (side / profile view)
IDEAL_SIDE_NECK_MAX = 15.0          # degrees forward from vertical (Ear over Shoulder)
IDEAL_SIDE_TORSO_MAX = 14.0         # degrees forward lean (Shoulder over Hip)
IDEAL_SIDE_CHIN_DROP_MAX = 16.0     # degrees head pitch downward when hip is out of frame
DEFAULT_SIDE_FACE_RATIO = 0.095     # ear-to-nose span / frame width

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


def _draw_dashed_vertical(
    canvas: np.ndarray,
    x: int,
    top_y: int,
    bottom_y: int,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw a subtle vertical plumb reference line for side-view posture assessment."""
    y_start = max(0, min(top_y, bottom_y))
    y_end = min(canvas.shape[0] - 1, max(top_y, bottom_y))
    dash_len = 7
    gap_len = 6
    y = y_start
    while y < y_end:
        y2 = min(y + dash_len, y_end)
        cv2.line(canvas, (x, y), (x, y2), color, thickness, cv2.LINE_AA)
        y += dash_len + gap_len


def _draw_side_skeleton(
    frame: np.ndarray,
    ear: Tuple[float, float],
    shoulder: Tuple[float, float],
    nose: Tuple[float, float],
    hip: Optional[Tuple[float, float]],
    status: str,
    neck_forward: bool,
    torso_slouched: bool,
    too_close: bool,
) -> np.ndarray:
    """Render minimalist clinical side-profile alignment lines in RAM."""
    annotated = frame.copy()
    overlay = frame.copy()

    pt_ear = (int(round(ear[0])), int(round(ear[1])))
    pt_sh = (int(round(shoulder[0])), int(round(shoulder[1])))
    pt_nose = (int(round(nose[0])), int(round(nose[1])))
    pt_hip = (int(round(hip[0])), int(round(hip[1]))) if hip is not None else None

    status_color = STATUS_BGR.get(status, BGR_POWDER)
    neck_color = STATUS_BGR["poor"] if neck_forward else BGR_MIST
    torso_color = STATUS_BGR["warning"] if torso_slouched else BGR_POWDER
    eye_color = STATUS_BGR["poor"] if too_close else BGR_POWDER

    # Vertical ideal plumb reference line rising from shoulder (or hip)
    plumb_top = max(10, pt_ear[1] - 35)
    plumb_bot = min(frame.shape[0] - 10, (pt_hip[1] if pt_hip else pt_sh[1]) + 25)
    _draw_dashed_vertical(overlay, pt_sh[0], plumb_top, plumb_bot, BGR_SNOW, thickness=1)

    # Soft Midnight Blue contrast backing
    if pt_hip is not None:
        cv2.line(overlay, pt_hip, pt_sh, BGR_MIDNIGHT, 6, cv2.LINE_AA)
    cv2.line(overlay, pt_sh, pt_ear, BGR_MIDNIGHT, 6, cv2.LINE_AA)
    cv2.line(overlay, pt_ear, pt_nose, BGR_MIDNIGHT, 4, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

    # Crisp primary profile axes
    if pt_hip is not None:
        cv2.line(annotated, pt_hip, pt_sh, torso_color, 3, cv2.LINE_AA)
        _draw_joint(annotated, pt_hip, torso_color, radius=6)
    cv2.line(annotated, pt_sh, pt_ear, neck_color, 3, cv2.LINE_AA)
    cv2.line(annotated, pt_ear, pt_nose, eye_color, 2, cv2.LINE_AA)

    _draw_joint(annotated, pt_sh, torso_color, radius=6)
    _draw_joint(annotated, pt_ear, neck_color, radius=6)
    _draw_joint(annotated, pt_nose, status_color, radius=5)

    return annotated


def _analyze_side_profile(
    frame: np.ndarray,
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
    baseline_neck_angle: Optional[float],
    baseline_torso_angle: Optional[float],
    baseline_face_ratio: Optional[float],
    include_image: bool,
) -> Dict[str, Any]:
    """Evaluate ergonomic posture from a left or right side profile."""
    frame_height, frame_width = frame.shape[:2]

    left_score = float(kps_conf[KP_LEFT_SHOULDER]) + (
        float(kps_conf[KP_LEFT_EAR]) if len(kps_conf) > KP_LEFT_EAR else 0.0
    )
    right_score = float(kps_conf[KP_RIGHT_SHOULDER]) + (
        float(kps_conf[KP_RIGHT_EAR]) if len(kps_conf) > KP_RIGHT_EAR else 0.0
    )
    use_left = left_score >= right_score
    side_facing = "left" if use_left else "right"

    idx_sh = KP_LEFT_SHOULDER if use_left else KP_RIGHT_SHOULDER
    idx_ear = KP_LEFT_EAR if use_left else KP_RIGHT_EAR
    idx_hip = KP_LEFT_HIP if use_left else KP_RIGHT_HIP

    sh_conf = float(kps_conf[idx_sh])
    ear_conf = float(kps_conf[idx_ear]) if len(kps_conf) > idx_ear else 0.0
    nose_conf = float(kps_conf[KP_NOSE])

    if sh_conf < 0.45 or (ear_conf < 0.40 and nose_conf < 0.40):
        return _undetected_response(frame, include_image=include_image, view_mode="side")

    shoulder = (float(kps_xy[idx_sh][0]), float(kps_xy[idx_sh][1]))
    nose = (float(kps_xy[KP_NOSE][0]), float(kps_xy[KP_NOSE][1]))
    ear = (
        (float(kps_xy[idx_ear][0]), float(kps_xy[idx_ear][1]))
        if ear_conf >= 0.40
        else nose
    )

    def _in_frame(pt: Tuple[float, float], margin: float = 0.02) -> bool:
        return (
            margin * frame_width <= pt[0] <= (1.0 - margin) * frame_width
            and margin * frame_height <= pt[1] <= (1.0 - margin) * frame_height
        )

    # Reject out-of-frame or upside-down glitch detections (still allow deep forward slouching)
    if not _in_frame(shoulder, 0.02) or not _in_frame(ear, 0.02):
        return _undetected_response(frame, include_image=include_image, view_mode="side")
    if ear[1] > shoulder[1] + 0.08 * frame_height:
        return _undetected_response(frame, include_image=include_image, view_mode="side")

    # Strict in-frame hip check to avoid phantom edge hips pulling the torso line
    hip: Optional[Tuple[float, float]] = None
    if len(kps_xy) > idx_hip and float(kps_conf[idx_hip]) >= 0.55:
        cand_hip = (float(kps_xy[idx_hip][0]), float(kps_xy[idx_hip][1]))
        if (
            0.05 * frame_width <= cand_hip[0] <= 0.95 * frame_width
            and cand_hip[1] <= 0.93 * frame_height
            and cand_hip[1] > shoulder[1] + max(20.0, 0.12 * frame_height)
        ):
            hip = cand_hip

    # Facing direction in image coordinates (+1 right, -1 left) — robust when looking down
    dir_signal = (nose[0] - ear[0]) + 0.55 * (nose[0] - shoulder[0])
    face_dir = 1.0 if dir_signal >= 0 else -1.0

    # 1. Forward Head / Neck Angle relative to vertical plumb line
    dy_neck = max(shoulder[1] - ear[1], 12.0)
    forward_dx = (ear[0] - shoulder[0]) * face_dir
    neck_angle = math.degrees(math.atan2(max(forward_dx, 0.0), dy_neck))
    target_neck_angle = _sanitize_baseline(baseline_neck_angle, low=0.0, high=35.0, default=0.0)
    effective_neck_excess = max(0.0, neck_angle - max(IDEAL_SIDE_NECK_MAX, target_neck_angle + 6.0))
    neck_forward = effective_neck_excess > 0.0
    if neck_angle > max(25.0, target_neck_angle + 15.0):
        neck_status = "Kepala Maju"
    elif neck_forward:
        neck_status = "Agak Maju"
    else:
        neck_status = "Lurus"

    # 2. Torso Lean (Shoulder over Hip) OR Chin Drop fallback if hip is outside frame
    if hip is not None:
        dy_torso = max(hip[1] - shoulder[1], 1.0)
        torso_angle = abs(math.degrees(math.atan2(abs(shoulder[0] - hip[0]), dy_torso)))
        target_torso = _sanitize_baseline(baseline_torso_angle, low=0.0, high=30.0, default=0.0)
        torso_limit = max(IDEAL_SIDE_TORSO_MAX, target_torso + 6.0)
        torso_slouched = torso_angle > torso_limit
        torso_label = "Membungkuk" if torso_angle > torso_limit + 10.0 else ("Agak Bungkuk" if torso_slouched else "Tegak")
        torso_metric_mode = "torso"
    else:
        dx_head = max(abs(nose[0] - ear[0]), 1.0)
        dy_drop = max(nose[1] - ear[1], 0.0)
        torso_angle = math.degrees(math.atan2(dy_drop, dx_head))
        target_torso = _sanitize_baseline(baseline_torso_angle, low=0.0, high=30.0, default=0.0)
        torso_limit = max(IDEAL_SIDE_CHIN_DROP_MAX, target_torso + 6.0)
        torso_slouched = torso_angle > torso_limit
        torso_label = "Menunduk" if torso_angle > torso_limit + 10.0 else ("Agak Menunduk" if torso_slouched else "Tegak")
        torso_metric_mode = "chin_drop"

    # 3. Side Screen Proximity (ear-to-nose scale / frame width)
    head_span = math.hypot(nose[0] - ear[0], nose[1] - ear[1])
    if head_span < 8.0:
        head_span = math.hypot(nose[0] - shoulder[0], nose[1] - shoulder[1]) * 0.45
    face_ratio = head_span / max(float(frame_width), 1.0)
    target_face = _sanitize_baseline(baseline_face_ratio, low=0.03, high=0.45, default=DEFAULT_SIDE_FACE_RATIO)
    face_increase = max(0.0, (face_ratio - target_face) / max(target_face, 1e-6))
    too_close = face_increase > FACE_INCREASE_THRESHOLD
    distance_status = "Terlalu Dekat" if too_close else ("Agak Dekat" if face_increase > 0.12 else "Aman")

    # 4. Ergonomic Scoring (0 - 100)
    if not neck_forward:
        neck_penalty = max(0.0, (neck_angle - 10.0) * 0.8)
    else:
        neck_penalty = 20.0 + min(30.0, effective_neck_excess * 2.2)

    if not torso_slouched:
        torso_penalty = max(0.0, (torso_angle - 9.0) * 0.8)
    else:
        torso_penalty = 16.0 + min(26.0, (torso_angle - torso_limit) * 2.2)

    if not too_close:
        face_penalty = max(0.0, (face_increase - 0.12) * 30.0)
    else:
        face_penalty = 16.0 + min(24.0, (face_increase - FACE_INCREASE_THRESHOLD) * 70.0)

    score = int(round(max(0.0, min(100.0, 100.0 - (neck_penalty + torso_penalty + face_penalty)))))
    if score >= 80:
        status = "ideal"
        status_label = "Postur Bagus & Sehat"
    elif score >= 55:
        status = "warning"
        status_label = "Perlu Diperbaiki Sedikit"
    else:
        status = "poor"
        status_label = "Postur Kurang Baik"

    issues: List[Dict[str, str]] = []
    if neck_forward:
        issues.append(
            {
                "type": "tech_neck",
                "severity": "high" if neck_angle > 25.0 else "medium",
                "title": "Kepala & Leher Condong ke Depan",
                "detail": f"Sudut leher dari samping {neck_angle:.1f}° ke depan (ideal ≤ {IDEAL_SIDE_NECK_MAX:.0f}°).",
                "recommendation": "Tarik kepala dan dagu sejajar di atas bahu, lalu posisikan layar setinggi pandangan mata.",
            }
        )
    if torso_slouched:
        issues.append(
            {
                "type": "shoulder_tilt",
                "severity": "high" if torso_angle > torso_limit + 10.0 else "medium",
                "title": "Punggung Membungkuk" if torso_metric_mode == "torso" else "Kepala Terlalu Menunduk",
                "detail": f"Kemiringan terukur {torso_angle:.1f}° (batas ideal ≤ {torso_limit:.0f}°).",
                "recommendation": "Tegakkan punggung bersandar nyaman di kursi dan hindari membungkuk ke meja.",
            }
        )
    if too_close:
        issues.append(
            {
                "type": "eye_strain",
                "severity": "high" if face_increase > 0.42 else "medium",
                "title": "Wajah Terlalu Dekat ke Layar",
                "detail": f"Posisi wajah {round(face_increase * 100)}% lebih dekat dari jarak aman.",
                "recommendation": "Mundurkan posisi duduk sekitar 50–70 cm agar mata tidak cepat lelah.",
            }
        )

    inv_w = 1.0 / max(float(frame_width), 1.0)
    inv_h = 1.0 / max(float(frame_height), 1.0)
    normalized_keypoints: Dict[str, Any] = {
        "mode": "side",
        "side": side_facing,
        "ear": [round(ear[0] * inv_w, 4), round(ear[1] * inv_h, 4)],
        "shoulder": [round(shoulder[0] * inv_w, 4), round(shoulder[1] * inv_h, 4)],
        "nose": [round(nose[0] * inv_w, 4), round(nose[1] * inv_h, 4)],
        "hip": [round(hip[0] * inv_w, 4), round(hip[1] * inv_h, 4)] if hip else None,
        "face_dir": face_dir,
        "flags": {
            "neck_forward": neck_forward,
            "torso_slouched": torso_slouched,
            "too_close": too_close,
        },
    }

    image_data_uri: Optional[str] = None
    if include_image:
        annotated = _draw_side_skeleton(
            frame=frame,
            ear=ear,
            shoulder=shoulder,
            nose=nose,
            hip=hip,
            status=status,
            neck_forward=neck_forward,
            torso_slouched=torso_slouched,
            too_close=too_close,
        )
        image_data_uri = _encode_jpeg_data_uri(annotated, quality=80)

    return {
        "detected": True,
        "view_mode": "side",
        "image": image_data_uri,
        "keypoints": normalized_keypoints,
        "score": score,
        "status": status,
        "status_label": status_label,
        "metrics": {
            "shoulder_tilt": round(torso_angle, 1),
            "torso_angle": round(torso_angle, 1),
            "torso_status": torso_label,
            "torso_metric_mode": torso_metric_mode,
            "neck_angle": round(neck_angle, 1),
            "neck_ratio": round(neck_angle, 1),
            "face_ratio": round(face_ratio, 3),
            "neck_status": neck_status,
            "distance_status": distance_status,
            "shoulder_status": torso_label,
        },
        "issues": issues,
    }


def _analyze_frame_sync(
    frame: np.ndarray,
    baseline_neck_ratio: Optional[float],
    baseline_face_ratio: Optional[float],
    include_image: bool = True,
    view_mode: str = "auto",
    baseline_side_neck_angle: Optional[float] = None,
    baseline_side_torso_angle: Optional[float] = None,
    baseline_side_face_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """Run YOLO11n-pose inference, auto/manual front or side ergonomic analysis, and optional drawing."""
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
        return _undetected_response(frame, include_image=include_image, view_mode=view_mode)

    kps_xy = results[0].keypoints.xy[0].cpu().numpy()
    kps_conf = (
        results[0].keypoints.conf[0].cpu().numpy()
        if results[0].keypoints.conf is not None
        else np.ones(len(kps_xy), dtype=np.float32)
    )

    if len(kps_xy) <= KP_RIGHT_SHOULDER:
        return _undetected_response(frame, include_image=include_image, view_mode=view_mode)

    mode_clean = (view_mode or "auto").lower().strip()
    if mode_clean not in {"auto", "front", "side"}:
        mode_clean = "auto"

    nose_conf = float(kps_conf[KP_NOSE])
    ls_conf = float(kps_conf[KP_LEFT_SHOULDER])
    rs_conf = float(kps_conf[KP_RIGHT_SHOULDER])

    # Determine whether to run Side Profile or Front View (slouch-proof Euclidean + nose-offset)
    use_side = mode_clean == "side"
    if mode_clean == "auto":
        both_sh_ok = ls_conf >= KEYPOINT_CONF_THRESHOLD and rs_conf >= KEYPOINT_CONF_THRESHOLD
        if not both_sh_ok and (ls_conf >= 0.40 or rs_conf >= 0.40):
            use_side = True
        elif both_sh_ok:
            ls_x, ls_y = float(kps_xy[KP_LEFT_SHOULDER][0]), float(kps_xy[KP_LEFT_SHOULDER][1])
            rs_x, rs_y = float(kps_xy[KP_RIGHT_SHOULDER][0]), float(kps_xy[KP_RIGHT_SHOULDER][1])
            nose_x, nose_y = float(kps_xy[KP_NOSE][0]), float(kps_xy[KP_NOSE][1])
            dx_sh = abs(ls_x - rs_x)
            mid_sh_x = (ls_x + rs_x) / 2.0
            mid_sh_y = (ls_y + rs_y) / 2.0
            # Euclidean neck-to-nose length stays large even during heavy forward slouching
            head_dist = max(math.hypot(mid_sh_x - nose_x, mid_sh_y - nose_y), 24.0)
            sh_to_head = dx_sh / head_dist
            nose_out = abs(nose_x - mid_sh_x) / max(dx_sh, 1.0)
            if sh_to_head < 0.78 or (nose_out > 0.85 and sh_to_head < 1.05):
                use_side = True

    if use_side:
        return _analyze_side_profile(
            frame=frame,
            kps_xy=kps_xy,
            kps_conf=kps_conf,
            baseline_neck_angle=baseline_side_neck_angle,
            baseline_torso_angle=baseline_side_torso_angle,
            baseline_face_ratio=baseline_side_face_ratio,
            include_image=include_image,
        )

    # Front View: Confidence validation for nose & both shoulders
    if (
        nose_conf < KEYPOINT_CONF_THRESHOLD
        or ls_conf < KEYPOINT_CONF_THRESHOLD
        or rs_conf < KEYPOINT_CONF_THRESHOLD
    ):
        return _undetected_response(frame, include_image=include_image, view_mode="front")

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
        "mode": "front",
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
        "view_mode": "front",
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


def _undetected_response(frame: np.ndarray, include_image: bool = True, view_mode: str = "front") -> Dict[str, Any]:
    """Return structured undetected payload when key joints are missing or low confidence."""
    is_side = view_mode == "side"
    msg = (
        "Arahkan posisi samping kepala dan bahu Anda terlihat jelas di kamera."
        if is_side
        else "Arahkan wajah dan kedua bahu Anda ke depan kamera di tempat yang cukup terang."
    )
    return {
        "detected": False,
        "view_mode": view_mode,
        "image": _encode_jpeg_data_uri(frame, quality=75) if include_image else None,
        "keypoints": None,
        "score": None,
        "status": "undetected",
        "status_label": "Tubuh Belum Terlihat Jelas",
        "message": msg,
        "metrics": None,
        "issues": [
            {
                "type": "detection",
                "severity": "info",
                "title": "Arahkan Tubuh ke Kamera",
                "detail": msg,
                "recommendation": "Pastikan area kepala hingga bahu (atau pinggul untuk tampak samping) terlihat penuh di tempat terang.",
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
    view_mode: str = Form(default="auto"),
    baseline_side_neck_angle: Optional[float] = Form(default=None),
    baseline_side_torso_angle: Optional[float] = Form(default=None),
    baseline_side_face_ratio: Optional[float] = Form(default=None),
):
    """Analyze a single webcam frame purely in RAM (front or side profile)."""
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
        view_mode,
        baseline_side_neck_angle,
        baseline_side_torso_angle,
        baseline_side_face_ratio,
    )
    return JSONResponse(content=result)
