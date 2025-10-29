from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


FRAMES_DIR = "frames"


@dataclass
class FrameRecord:
    frame_number: int
    frame_path: str
    frame_url: str
    metrics: Dict[str, float]
    video_score: float
    photo_score: float


def _ensure_frames_dir() -> None:
    os.makedirs(FRAMES_DIR, exist_ok=True)


def _host() -> str:
    host = os.getenv("RENDER_EXTERNAL_HOSTNAME", "").strip()
    if host:
        scheme = "https" if host.endswith("onrender.com") else "http"
        return f"{scheme}://{host}"
    return "http://127.0.0.1:8000"


def _frame_url(filename: str) -> str:
    return f"{_host()}/frames/{filename}"


app = FastAPI(
    title="Framesque Backdrop Coach",
    description=(
        "Analyzes video frames to score backdrops for content creation, "
        "selects the best options for video and photography, and returns "
        "pose coaching informed by scene metrics and MediaPipe landmarks."
    ),
    version="2.0.0",
)

_ensure_frames_dir()
app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")

mp_pose = mp.solutions.pose


@app.get("/")
def home():
    return {"message": "Framesque Backdrop Coach is running"}


def _normalize(value: float, target: float, tolerance: float) -> float:
    span = max(tolerance, 1e-6)
    score = 1.0 - min(1.0, abs(value - target) / span)
    return max(0.0, min(1.0, score))


def _compute_colorfulness(frame: np.ndarray) -> float:
    b, g, r = cv2.split(frame.astype("float"))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    rg_mean, rg_std = np.mean(rg), np.std(rg)
    yb_mean, yb_std = np.mean(yb), np.std(yb)
    colorfulness = math.sqrt(rg_std**2 + yb_std**2) + 0.3 * math.sqrt(rg_mean**2 + yb_mean**2)
    return float(np.tanh(colorfulness / 100.0))


def _compute_metrics(frame: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray) / 255.0)
    contrast = float(np.std(gray) / 128.0)
    saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
    edges = cv2.Canny(gray, 100, 200)
    clutter = float(np.mean(edges) / 255.0)
    colorfulness = _compute_colorfulness(frame)

    metrics = {
        "brightness": float(np.clip(brightness, 0.0, 1.0)),
        "contrast": float(np.clip(contrast, 0.0, 1.0)),
        "saturation": float(np.clip(saturation, 0.0, 1.0)),
        "clutter": float(np.clip(clutter, 0.0, 1.0)),
        "colorfulness": float(np.clip(colorfulness, 0.0, 1.0)),
    }
    return metrics


def _score_for_video(metrics: Dict[str, float]) -> float:
    exposure = _normalize(metrics["brightness"], target=0.6, tolerance=0.25)
    texture = _normalize(metrics["contrast"], target=0.45, tolerance=0.3)
    color = (metrics["saturation"] + metrics["colorfulness"]) / 2
    openness = 1.0 - metrics["clutter"]

    score = 0.34 * exposure + 0.28 * texture + 0.2 * color + 0.18 * openness
    return float(np.clip(score, 0.0, 1.0))


def _score_for_photo(metrics: Dict[str, float]) -> float:
    glow = _normalize(metrics["brightness"], target=0.55, tolerance=0.2)
    palette = _normalize(metrics["colorfulness"], target=0.65, tolerance=0.25)
    saturation = _normalize(metrics["saturation"], target=0.55, tolerance=0.25)
    tidy = 1.0 - metrics["clutter"]

    score = 0.3 * glow + 0.25 * palette + 0.25 * saturation + 0.2 * tidy
    return float(np.clip(score, 0.0, 1.0))


def _pose_observation(image_path: str) -> Dict[str, float]:
    image = cv2.imread(image_path)
    if image is None:
        return {"detected": False, "error": "Unable to load frame for pose analysis."}

    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return {"detected": False, "error": "No subject detected. Step into frame for pose coaching."}

    h, w = image.shape[:2]
    landmarks = results.pose_landmarks.landmark

    def _xy(index: mp_pose.PoseLandmark) -> np.ndarray:
        lm = landmarks[index.value]
        return np.array([lm.x * w, lm.y * h])

    left_shoulder = _xy(mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_shoulder = _xy(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    left_hip = _xy(mp_pose.PoseLandmark.LEFT_HIP)
    right_hip = _xy(mp_pose.PoseLandmark.RIGHT_HIP)
    left_ankle = _xy(mp_pose.PoseLandmark.LEFT_ANKLE)
    right_ankle = _xy(mp_pose.PoseLandmark.RIGHT_ANKLE)

    shoulder_vec = left_shoulder - right_shoulder
    hip_vec = left_hip - right_hip
    stance_vec = left_ankle - right_ankle

    shoulder_tilt = float(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0])))
    hip_tilt = float(np.degrees(np.arctan2(hip_vec[1], hip_vec[0])))
    stance_width = float(np.linalg.norm(stance_vec) / max(w, 1))
    shoulder_width = float(np.linalg.norm(shoulder_vec) / max(w, 1))
    stance_ratio = stance_width / max(shoulder_width, 1e-6)

    chest_forward = float(np.clip(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility, 0.0, 1.0) +
                          np.clip(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility, 0.0, 1.0)) / 2

    return {
        "detected": True,
        "shoulder_tilt": shoulder_tilt,
        "hip_tilt": hip_tilt,
        "stance_ratio": stance_ratio,
        "chest_forward": chest_forward,
    }


def _pose_tips(metrics: Dict[str, float], context: str, observation: Dict[str, float] | None) -> List[str]:
    tips: List[str] = []

    brightness = metrics["brightness"]
    clutter = metrics["clutter"]
    colorfulness = metrics["colorfulness"]
    contrast = metrics["contrast"]

    if brightness < 0.45:
        tips.append("Angle your face toward the brightest source so your eyes catch more light.")
    elif brightness > 0.75:
        tips.append("Step back half a pace or add diffusion to soften the highlights on your skin.")
    else:
        tips.append("Lighting is balanced—stay roughly where you are relative to the key light.")

    if clutter > 0.35:
        tips.append("Create separation: stand a step forward or shoot with a wider aperture to blur the backdrop.")
    else:
        tips.append("Lean into the clean lines—center yourself and let the backdrop frame you.")

    if colorfulness > 0.6:
        tips.append("Keep wardrobe neutral so the colorful backdrop remains the hero.")
    else:
        tips.append("Add a pop of color (jacket, lipstick, prop) to energize the scene.")

    if context == "video":
        tips.append("Keep gestures in the middle third of the frame so they stay inside safe broadcast margins.")
    else:
        tips.append("Shift weight onto your back foot and drop the front shoulder for a relaxed S-curve.")

    if observation is None:
        return tips

    if not observation.get("detected"):
        error = observation.get("error")
        if error:
            tips.append(error)
        else:
            tips.append("Step fully into frame so the pose coach can fine-tune your posture.")
        return tips

    shoulder_tilt = abs(observation.get("shoulder_tilt", 0.0))
    hip_tilt = abs(observation.get("hip_tilt", 0.0))
    stance_ratio = observation.get("stance_ratio", 0.0)
    chest_forward = observation.get("chest_forward", 0.0)

    if shoulder_tilt > 8:
        tips.append("Drop the higher shoulder and lengthen through your spine to level your posture.")
    else:
        tips.append("Your shoulders are level—keep that relaxed height for confidence.")

    if hip_tilt > 10:
        tips.append("Square your hips to camera so the stance reads grounded.")
    else:
        tips.append("Hips are aligned—feel free to add a subtle angle for shape.")

    if stance_ratio < 0.9:
        tips.append("Widen your stance slightly so your base feels stable on camera.")
    elif stance_ratio > 1.4:
        tips.append("Bring your feet a touch closer together for a refined silhouette.")
    else:
        tips.append("Stance width looks balanced—keep the weight soft between both feet.")

    if context == "video":
        if chest_forward < 0.5:
            tips.append("Open your chest toward the lens so your voice projects clearly.")
        else:
            tips.append("Great openness—keep breathing and use gentle forward energy when speaking.")
    else:
        if contrast > 0.5:
            tips.append("Let your chin glide forward and down slightly to define the jawline.")
        else:
            tips.append("Lift through the crown and relax the jaw for a softer portrait feel.")

    return tips


@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...)):
    _ensure_frames_dir()

    temp_filename = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())

    cap = cv2.VideoCapture(temp_filename)
    frame_records: List[FrameRecord] = []

    if not cap.isOpened():
        os.remove(temp_filename)
        return JSONResponse({"error": "Unable to open video file"}, status_code=400)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_index % 30 == 0:
            filename = f"frame_{uuid.uuid4().hex[:8]}.jpg"
            frame_path = os.path.join(FRAMES_DIR, filename)
            cv2.imwrite(frame_path, frame)

            metrics = _compute_metrics(frame)
            record = FrameRecord(
                frame_number=frame_index,
                frame_path=frame_path,
                frame_url=_frame_url(filename),
                metrics=metrics,
                video_score=_score_for_video(metrics),
                photo_score=_score_for_photo(metrics),
            )
            frame_records.append(record)

    cap.release()
    os.remove(temp_filename)

    if not frame_records:
        return JSONResponse({"message": "No frames extracted — try a longer video"}, status_code=400)

    frame_records.sort(key=lambda rec: rec.frame_number)

    best_video = max(frame_records, key=lambda rec: rec.video_score)
    best_photo = max(frame_records, key=lambda rec: rec.photo_score)

    video_pose = _pose_observation(best_video.frame_path)
    photo_pose = _pose_observation(best_photo.frame_path)

    analysis = [
        {
            "frame_number": rec.frame_number,
            "frame_url": rec.frame_url,
            "metrics": rec.metrics,
            "video_score": rec.video_score,
            "photo_score": rec.photo_score,
        }
        for rec in frame_records[:20]
    ]

    response = {
        "frames": [rec.frame_url for rec in frame_records[:10]],
        "analysis": analysis,
        "best_backdrops": {
            "video": {
                "frame_number": best_video.frame_number,
                "frame_url": best_video.frame_url,
                "metrics": best_video.metrics,
                "score": best_video.video_score,
            },
            "photography": {
                "frame_number": best_photo.frame_number,
                "frame_url": best_photo.frame_url,
                "metrics": best_photo.metrics,
                "score": best_photo.photo_score,
            },
        },
        "pose_guidance": {
            "video": _pose_tips(best_video.metrics, "video", video_pose),
            "photography": _pose_tips(best_photo.metrics, "photo", photo_pose),
        },
    }

    return response
