from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
import uuid
from PIL import Image
import torch
from aesthetics_predictor import AestheticsPredictorV1
from transformers import CLIPProcessor

# -------------------------------------------------------
# Setup
# -------------------------------------------------------
def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

_ensure_dir("frames")

app = FastAPI(
    title="Framesque AI Mentor",
    description="Analyzes real scenes and teaches creators to shoot like pros — now with neural aesthetic scoring.",
    version="3.5.0",
)

app.mount("/frames", StaticFiles(directory="frames"), name="frames")

@app.get("/")
def home():
    return {"message": "Framesque v3.5 is running with AI aesthetic analysis!"}

# -------------------------------------------------------
# Load the Aesthetic Model
# -------------------------------------------------------
print("Loading aesthetic model... this may take a moment ⏳")
model_id = "shunk031/aesthetics-predictor-v1-vit-base-patch16"
processor = CLIPProcessor.from_pretrained(model_id)
aesthetic_model = AestheticsPredictorV1.from_pretrained(model_id)
aesthetic_model.eval()
print("✅ Aesthetic model loaded successfully.")

def compute_aesthetic_score(image_path: str) -> float:
    """Compute AI-based aesthetic score using CLIP embeddings."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = aesthetic_model(**inputs)
        if hasattr(outputs, "logits"):
            score = outputs.logits.squeeze().item()
        else:
            score = float(outputs[0])  # fallback
    return round(float(score), 3)

# -------------------------------------------------------
# Visual metric helpers (for lighting + composition)
# -------------------------------------------------------
def _normalize(values):
    if not values:
        return []
    min_v, max_v = min(values), max(values)
    if abs(max_v - min_v) < 1e-6:
        return [0.5 for _ in values]
    return [(v - min_v) / (max_v - min_v) for v in values]

def _analyze_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(gray) / 255.0)
    contrast = float(np.std(gray) / 255.0)
    saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
    edges = cv2.Canny(gray, 100, 200)
    clutter = float(np.mean(edges) / 255.0)
    return {"brightness": brightness, "contrast": contrast, "saturation": saturation, "clutter": clutter}

# -------------------------------------------------------
# Mentorship text generator
# -------------------------------------------------------
def _compose_guidance(metrics, aesthetic_score):
    tips = []
    brightness, contrast, sat, clutter = metrics.values()

    if brightness < 0.45:
        tips.append("Step closer to a window or add soft light to brighten exposure.")
    elif brightness > 0.75:
        tips.append("Angle slightly away from the light to reduce highlights.")
    else:
        tips.append("Exposure looks balanced — maintain your current distance from the light.")

    if clutter > 0.35:
        tips.append("Keep some distance from the background or use wider aperture to create separation.")
    else:
        tips.append("The space feels clean — good for minimalist editorials.")

    if contrast > 0.4 and sat > 0.4:
        tips.append("Lean into the punchy contrast; confident posture and crisp lines will complement it.")
    else:
        tips.append("Go for soft posing; relaxed shoulders and subtle expressions will suit this tone.")

    if aesthetic_score >= 0.75:
        tips.append("This frame has strong editorial appeal — consider a 45° camera angle and natural light.")
    elif aesthetic_score >= 0.6:
        tips.append("Good aesthetic balance; experiment with subtle motion or prop interaction.")
    else:
        tips.append("Lighting is serviceable but not ideal — add side lighting or simplify composition.")

    return tips

# -------------------------------------------------------
# Core API Endpoint
# -------------------------------------------------------
@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze video for editorial-quality backdrops with AI guidance."""
    temp_video = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_video, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(temp_video)
    if not cap.isOpened():
        return JSONResponse({"error": "Could not open video."}, status_code=400)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_num % 30 == 0:
            frame_name = f"frame_{uuid.uuid4().hex[:8]}.jpg"
            frame_path = os.path.join("frames", frame_name)
            cv2.imwrite(frame_path, frame)
            metrics = _analyze_frame(frame)
            aesthetic_score = compute_aesthetic_score(frame_path)
            frames.append({
                "frame_number": frame_num,
                "frame_url": f"https://framesque-backend.onrender.com/frames/{frame_name}",
                "metrics": metrics,
                "aesthetic_score": aesthetic_score
            })

    cap.release()
    os.remove(temp_video)

    if not frames:
        return JSONResponse({"message": "No frames extracted — try a longer video"}, status_code=400)

    norm_aesthetic = _normalize([f["aesthetic_score"] for f in frames])
    norm_bright = _normalize([f["metrics"]["brightness"] for f in frames])
    norm_contrast = _normalize([f["metrics"]["contrast"] for f in frames])
    norm_sat = _normalize([f["metrics"]["saturation"] for f in frames])
    norm_clutter = _normalize([f["metrics"]["clutter"] for f in frames])

    composite_scores = []
    for i, f in enumerate(frames):
        score = (
            0.5 * norm_aesthetic[i]
            + 0.2 * norm_contrast[i]
            + 0.2 * norm_sat[i]
            + 0.1 * (1 - norm_clutter[i])
        )
        composite_scores.append(score)

    best_idx = int(np.argmax(composite_scores))
    best_frame = frames[best_idx]

    return {
        "best_frame": {
            "frame_url": best_frame["frame_url"],
            "aesthetic_score": best_frame["aesthetic_score"],
            "metrics": best_frame["metrics"],
            "overall_score": composite_scores[best_idx],
            "guidance": _compose_guidance(best_frame["metrics"], best_frame["aesthetic_score"]),
        },
        "summary": "This backdrop scored highest for editorial balance and visual appeal. Follow the guidance to optimize your light, pose, and composition."
    }
