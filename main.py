from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
import uuid
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# -------------------------------------------------------
# Setup
# -------------------------------------------------------
def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

_ensure_dir("frames")

app = FastAPI(
    title="Framesque AI Mentor",
    description="Analyzes real spaces and teaches creators to shoot like pros — now with CLIP-based aesthetic vision.",
    version="3.6.0",
)

app.mount("/frames", StaticFiles(directory="frames"), name="frames")

@app.get("/")
def home():
    return {"message": "Framesque v3.6 is running with CLIP-based AI analysis!"}

# -------------------------------------------------------
# Load CLIP model (AI aesthetic analysis)
# -------------------------------------------------------
print("⏳ Loading CLIP model for AI aesthetic analysis...")
model_id = "openai/clip-vit-base-patch16"
processor = CLIPProcessor.from_pretrained(model_id)
clip_model = CLIPModel.from_pretrained(model_id)
clip_model.eval()
print("✅ CLIP model loaded successfully!")

def compute_aesthetic_score(image_path: str) -> float:
    """Use CLIP to measure similarity between image and high-quality editorial prompts."""
    image = Image.open(image_path).convert("RGB")
    prompts = [
        "editorial photoshoot, cinematic composition, professional lighting",
        "award-winning photograph, depth, color harmony, leading lines",
        "magazine portrait, soft natural light, creative composition"
    ]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        # similarity between image and text embeddings
        scores = outputs.logits_per_image.softmax(dim=1)
        avg_score = torch.mean(scores).item()
    return round(float(avg_score), 3)

# -------------------------------------------------------
# Visual metric helpers
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
# Mentor guidance generator
# -------------------------------------------------------
def _compose_guidance(metrics, aesthetic_score):
    tips = []
    b, c, s, cl = metrics.values()

    if b < 0.45:
        tips.append("Step closer to a window or soft light to boost exposure.")
    elif b > 0.75:
        tips.append("Move slightly away from the light source to reduce overexposure.")
    else:
        tips.append("Exposure is well-balanced — ideal for natural tones.")

    if cl > 0.35:
        tips.append("Increase subject-background distance or use a wider aperture for depth separation.")
    else:
        tips.append("The background is clean — perfect for editorial or portrait shots.")

    if c > 0.4 and s > 0.4:
        tips.append("Use confident posture and sharper framing — this lighting supports strong contrast.")
    else:
        tips.append("Go for relaxed posing and gentle light diffusion for a softer vibe.")

    if aesthetic_score >= 0.75:
        tips.append("This frame has strong editorial appeal. Consider shooting at a 45° angle with natural side lighting.")
    elif aesthetic_score >= 0.6:
        tips.append("Good potential — enhance it with subtle subject movement or prop composition.")
    else:
        tips.append("Scene feels flat. Try introducing side light or reduce visual clutter for stronger impact.")

    return tips

# -------------------------------------------------------
# Main API endpoint
# -------------------------------------------------------
@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze uploaded video and provide intelligent photography guidance."""
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
    norm_contrast = _normalize([f["metrics"]["contrast"] for f in frames])
    norm_sat = _normalize([f["metrics"]["saturation"] for f in frames])
    norm_clutter = _normalize([f["metrics"]["clutter"] for f in frames])

    scores = []
    for i, f in enumerate(frames):
        total = (
            0.5 * norm_aesthetic[i] +
            0.25 * norm_contrast[i] +
            0.15 * norm_sat[i] +
            0.10 * (1 - norm_clutter[i])
        )
        scores.append(total)

    best = frames[int(np.argmax(scores))]

    return {
        "best_frame": {
            "frame_url": best["frame_url"],
            "aesthetic_score": best["aesthetic_score"],
            "metrics": best["metrics"],
            "overall_score": max(scores),
            "guidance": _compose_guidance(best["metrics"], best["aesthetic_score"]),
        },
        "summary": "This frame scored highest in composition and editorial balance. Follow these mentoring tips to achieve professional-grade results."
    }
