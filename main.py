from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os, uuid, cv2, numpy as np, hashlib, math
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import CLIPProcessor, CLIPModel

# ---------- Optional VLM (GPT-4o) ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL   = os.getenv("OPENAI_VLM_MODEL", "gpt-4o-mini")

# ---------- Pose Estimation (MediaPipe) ----------
import mediapipe as mp
mp_pose    = mp.solutions.pose
POSE_LMS   = mp_pose.PoseLandmark
POSE_CONNS = mp_pose.POSE_CONNECTIONS

# ---------- Setup ----------
def _ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p)

_ensure_dir("frames")
_ensure_dir("frames/pose")
_ensure_dir("frames/preview")

app = FastAPI(
    title="Framesque Pro",
    description="Editorial backdrop selector + photography mentor (CLIP + GPT-4o + Transparent Pose Coach + Composited Preview).",
    version="5.6.0",
)

app.mount("/frames", StaticFiles(directory="frames"), name="frames")

@app.get("/")
def home():
    return {"message": "Framesque Pro v5.6 (CLIP + GPT-4o + Pose Coach + Composited Preview)."}

@app.get("/health")
def health():
    return {"status": "ok", "version": "5.6.0"}

# ---------- Host/URL helpers ----------
def _host():
    return os.getenv("RENDER_EXTERNAL_HOSTNAME", "localhost:8000")

def _abs_url(path: str) -> str:
    host = _host()
    scheme = "https" if "onrender.com" in host else "http"
    return f"{scheme}://{host}/{path.lstrip('/')}"

# ---------- CLIP (neural aesthetic proxy) ----------
print("⏳ Loading CLIP model...")
CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
clip_model.eval()
print("✅ CLIP ready.")

_aes_cache = {}
def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def compute_aesthetic_score(image_path: str) -> float:
    key = _file_md5(image_path)
    if key in _aes_cache:
        return _aes_cache[key]
    image = Image.open(image_path).convert("RGB")
    prompts = [
        "editorial photoshoot, cinematic composition, professional lighting",
        "award-winning photograph, depth, color harmony, leading lines",
        "magazine portrait, soft natural light, creative composition",
    ]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = clip_model(**inputs)
        score = out.logits_per_image.softmax(dim=1).mean().item()
    score = round(float(score), 3)
    _aes_cache[key] = score
    return score

# ---------- Low-level visual metrics ----------
def _normalize(vals):
    if not vals: return []
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-6: return [0.5 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]

def _analyze_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(gray) / 255.0)
    contrast   = float(np.std(gray) / 255.0)
    saturation = float(np.mean(hsv[:,:,1]) / 255.0)
    edges      = cv2.Canny(gray, 100, 200)
    clutter    = float(np.mean(edges) / 255.0)
    return {"brightness": brightness, "contrast": contrast, "saturation": saturation, "clutter": clutter}

# ---------- Rule-based mentoring (always available) ----------
def _compose_guidance(metrics, aesthetic_score):
    tips = []
    b, c, s, cl = metrics["brightness"], metrics["contrast"], metrics["saturation"], metrics["clutter"]

    if b < 0.45: tips.append("Boost exposure: face a window or bring a soft key light closer.")
    elif b > 0.75: tips.append("Reduce highlights: angle slightly off the light or add diffusion.")
    else: tips.append("Exposure is balanced — keep your current distance to the light.")

    if cl > 0.35: tips.append("Increase subject–background distance or use a wider aperture for separation.")
    else: tips.append("Backdrop is clean — stand closer for a graphic editorial look.")

    if c > 0.4 and s > 0.4: tips.append("Use confident posture and crisp framing — bold contrast suits it.")
    else: tips.append("Use relaxed posing and gentle angles — the scene reads softer.")

    if aesthetic_score >= 0.75: tips.append("Strong editorial potential — try a 45° camera angle with natural side light.")
    elif aesthetic_score >= 0.6: tips.append("Good baseline — elevate with subtle motion or a simple prop.")
    else: tips.append("Feels a bit flat — add side light or simplify the frame for impact.")

    return tips

# ---------- Optional GPT-4o critique (natural language) ----------
def _vlm_available() -> bool:
    return bool(OPENAI_API_KEY)

def _vlm_critique(image_url: str, metrics: dict, aesthetic_score: float) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "You are an experienced editorial photographer and creative director. "
            "Analyze this frame as a potential backdrop for a portrait/editorial shoot. "
            "Use the numeric metrics as technical context (brightness, contrast, saturation, clutter). "
            "Write a concise, expert critique with concrete instructions — no fluff, no repetition. "
            "Cover: why the backdrop works, exact camera placement/angle, subject placement & posing, "
            "lighting guidance (natural or artificial), and one quick set tweak to elevate the shot. "
            "Keep it under 120 words.\n\n"
            f"Metrics JSON: {metrics}\nCLIP_aesthetic_score: {aesthetic_score}\n"
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You give concise, authoritative photo direction."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

# ---------- Pose Coach (overlay + feedback + composition preview) ----------
def _angle_deg(p1, p2):
    dx, dy = (p2[0]-p1[0]), (p2[1]-p1[1])
    return math.degrees(math.atan2(dy, dx))

def _mid(p1, p2):
    return ((p1[0]+p2[0]) / 2.0, (p1[1]+p2[1]) / 2.0)

def _golden_grid(draw: ImageDraw.ImageDraw, w: int, h: int, alpha: int = 38):
    # Faint golden ratio lines
    phi = 1.618
    x1 = int(w / phi); x2 = int(w - w / phi)
    y1 = int(h / phi); y2 = int(h - h / phi)
    color = (255, 255, 255, alpha)
    for x in (x1, x2):
        draw.line([(x,0),(x,h)], fill=color, width=1)
    for y in (y1, y2):
        draw.line([(0,y),(w,y)], fill=color, width=1)

def _vignette_and_tone(base: Image.Image) -> Image.Image:
    # Mild editorial pop: soft contrast + vignette
    w, h = base.size
    # Contrast curve
    arr = np.array(base).astype(np.float32)
    arr = np.clip((arr - 127.5) * 1.06 + 127.5, 0, 255)

    # Vignette
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_norm = r / r.max()
    vignette = 0.15 * (r_norm ** 1.8)  # up to -15%
    if arr.ndim == 3:
        arr *= (1 - vignette[..., None])
    else:
        arr *= (1 - vignette)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def _watermark(draw: ImageDraw.ImageDraw, w: int, h: int, text: str = "Framesque"):
    font = ImageFont.load_default()
    tw, th = draw.textlength(text, font=font), 12  # approx height for default font
    pad = 12
    x = w - int(tw) - pad
    y = h - th - pad
    # subtle gray, 30% opacity
    draw.text((x, y), text, fill=(255, 255, 255, 76), font=font)

def _pose_overlay_and_feedback(image_path: str):
    """Return (overlay_path PNG RGBA, composited_preview_path JPG, feedback list)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, None, ["Could not read the best frame for pose analysis."]
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        results = pose.process(img_rgb)

    if not results.pose_landmarks:
        return None, None, ["No person detected for pose coaching. Step into frame and try again."]

    lms = results.pose_landmarks.landmark
    def P(lm): return (int(lms[lm].x * w), int(lms[lm].y * h))

    # Key points
    L_SH = P(POSE_LMS.LEFT_SHOULDER)
    R_SH = P(POSE_LMS.RIGHT_SHOULDER)
    L_HIP = P(POSE_LMS.LEFT_HIP)
    R_HIP = P(POSE_LMS.RIGHT_HIP)
    L_EAR = P(POSE_LMS.LEFT_EAR)
    R_EAR = P(POSE_LMS.RIGHT_EAR)

    # Angles
    shoulder_angle = _angle_deg(L_SH, R_SH)          # ~0 ideal
    hip_angle      = _angle_deg(L_HIP, R_HIP)        # ~0 ideal
    torso_angle    = _angle_deg(_mid(L_HIP, R_HIP), _mid(L_SH, R_SH))  # ~ -90 upright (y-down)
    head_tilt      = _angle_deg(L_EAR, R_EAR) if (L_EAR != (0,0) and R_EAR != (0,0)) else 0.0

    # Feedback
    fb = []
    if abs(shoulder_angle) > 7:
        fb.append("Level your shoulders slightly for symmetry.")
    else:
        fb.append("Shoulders look level — great base posture.")

    if abs(hip_angle) > 7:
        fb.append("Square your hips a touch for stability.")
    else:
        fb.append("Hips are balanced — stable stance.")

    if abs(abs(torso_angle) - 90) <= 8:
        fb.append("Torso alignment is upright — clean editorial silhouette.")
    else:
        fb.append("Add gentle contrapposto: shift weight to back foot, turn torso ~10–15°.")

    if abs(head_tilt) > 8:
        fb.append("Reduce head tilt a little for balance.")
    else:
        fb.append("Head alignment looks balanced.")

    # Transparent overlay
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_alpha = 110
    joint_alpha = 130
    color = (255, 255, 255, line_alpha)
    dot_color = (255, 255, 255, joint_alpha)
    radius = max(2, int(min(w, h) * 0.004))

    # Build pixel coords dict
    def safe_pt(lm):
        return (int(lms[lm].x * w), int(lms[lm].y * h))
    pts = {lm: safe_pt(lm) for lm in POSE_LMS}

    # Draw MediaPipe skeleton
    for a_idx, b_idx in POSE_CONNS:
        a = pts.get(POSE_LMS(a_idx))
        b = pts.get(POSE_LMS(b_idx))
        if a and b:
            draw.line([a, b], fill=color, width=2)

    # Emphasize shoulders/hips/spine
    draw.line([L_SH, R_SH], fill=(255, 255, 255, 150), width=3)
    draw.line([L_HIP, R_HIP], fill=(255, 255, 255, 150), width=3)
    draw.line([_mid(L_HIP, R_HIP), _mid(L_SH, R_SH)], fill=(255, 255, 255, 110), width=2)

    for key in [POSE_LMS.LEFT_SHOULDER, POSE_LMS.RIGHT_SHOULDER, POSE_LMS.LEFT_HIP, POSE_LMS.RIGHT_HIP]:
        x, y = pts[key]
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=dot_color, width=2)

    # Save transparent overlay
    base = os.path.splitext(os.path.basename(image_path))[0]
    overlay_name = f"pose_{base}.png"
    overlay_path = os.path.join("frames/pose", overlay_name)
    overlay.save(overlay_path, format="PNG")

    # Composited preview: base frame + overlay + golden grid + vignette + watermark
    base_img = Image.open(image_path).convert("RGB")
    preview = base_img.copy().convert("RGBA")
    # apply transparent overlay
    preview.alpha_composite(overlay)

    # faint golden-ratio grid
    grid_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    grid_draw = ImageDraw.Draw(grid_layer)
    _golden_grid(grid_draw, w, h, alpha=38)
    preview.alpha_composite(grid_layer)

    # watermark (bottom-right)
    wm_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    wm_draw = ImageDraw.Draw(wm_layer)
    _watermark(wm_draw, w, h, text="Framesque")
    preview.alpha_composite(wm_layer)

    # Editorial filter (tone + vignette), then save as JPEG
    preview_rgb = _vignette_and_tone(preview.convert("RGB"))
    composite_name = f"posecomposite_{base}.jpg"
    composite_path = os.path.join("frames/preview", composite_name)
    preview_rgb.save(composite_path, format="JPEG", quality=92, optimize=True)

    return overlay_path, composite_path, fb

# ---------- Main endpoint ----------
@app.post("/analyze/")
async def analyze_video(
    file: UploadFile = File(...),
    stride_seconds: float = Query(1.0, ge=0.2, le=3.0, description="Sampling stride in seconds (default 1s)"),
    max_frames: int = Query(20, ge=3, le=60, description="Max frames to analyze (default 20)"),
):
    """Analyze video, pick best editorial backdrop, add pose overlay + coaching & composited preview, and mentor the user."""
    temp_video = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_video, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(temp_video)
    if not cap.isOpened():
        return JSONResponse({"error": "Could not open video."}, status_code=400)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(fps * stride_seconds), 1)

    frames = []
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_idx % step == 0:
            fname = f"frame_{uuid.uuid4().hex[:8]}.jpg"
            fpath = os.path.join("frames", fname)
            cv2.imwrite(fpath, frame)

            metrics = _analyze_frame(frame)
            aest    = compute_aesthetic_score(fpath)

            frames.append({
                "frame_number": frame_idx,
                "frame_url": _abs_url(f"frames/{fname}"),
                "metrics": metrics,
                "aesthetic_score": aest,
                "disk_path": fpath,
            })
            saved += 1
            if saved >= max_frames:
                break

    cap.release()
    os.remove(temp_video)

    if not frames:
        return JSONResponse({"message": "No frames extracted — try a longer video"}, status_code=400)

    # Rank by composite score (weighted toward CLIP aesthetic)
    norm_aes = _normalize([f["aesthetic_score"] for f in frames])
    norm_con = _normalize([f["metrics"]["contrast"]   for f in frames])
    norm_sat = _normalize([f["metrics"]["saturation"] for f in frames])
    norm_clu = _normalize([f["metrics"]["clutter"]    for f in frames])

    comps = []
    for i, f in enumerate(frames):
        comps.append(0.58 * norm_aes[i] + 0.24 * norm_con[i] + 0.10 * norm_sat[i] + 0.08 * (1 - norm_clu[i]))

    best = frames[int(np.argmax(comps))]

    # Pose Coach outputs
    overlay_path, composite_path, pose_feedback = _pose_overlay_and_feedback(best["disk_path"])
    overlay_url    = _abs_url(overlay_path) if overlay_path else "No pose overlay generated."
    composited_url = _abs_url(composite_path) if composite_path else "No composited preview generated."

    # Mentoring text
    tips = _compose_guidance(best["metrics"], best["aesthetic_score"])
    critique = ""
    if OPENAI_API_KEY:
        critique = _vlm_critique(best["frame_url"], best["metrics"], best["aesthetic_score"])

    return {
        "best_frame": {
            "frame_url": best["frame_url"],
            "aesthetic_score": best["aesthetic_score"],
            "metrics": best["metrics"],
            "overall_score": max(comps),
        },
        "pose_coach": {
            "overlay_url": overlay_url,               # transparent PNG
            "composited_url": composited_url,         # merged JPEG preview with grid & watermark
            "feedback": pose_feedback,
        },
        "mentorship": {
            "tips": tips,
            "critique": critique or "Set OPENAI_API_KEY to enable GPT-4o critique."
        },
        "params": {"stride_seconds": stride_seconds, "max_frames": max_frames}
    }
