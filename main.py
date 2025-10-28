from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import uuid
import os
import cv2
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime

# -------------------- App Setup --------------------
app = FastAPI(
    title="Framesque API",
    description="AI-powered background discovery for content creators",
    version="1.1.0"
)

# Allow all CORS origins (for frontend use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving (so frame images can be viewed in browser)
app.mount("/frames", StaticFiles(directory="frames"), name="frames")

# Directories
UPLOAD_DIR = "uploads"
FRAMES_DIR = "frames"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# In-memory storage
videos = {}
frames = {}

# -------------------- Load AI Model --------------------
print("Loading CLIP model... (this may take a few seconds)")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP model loaded successfully!")

# -------------------- Helper Functions --------------------
def extract_frames(video_path: str, output_dir: str, num_frames: int = 10):
    """Extract evenly spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)
    frame_paths = []

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if not success:
            break

        frame_filename = f"frame_{uuid.uuid4().hex}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        if len(frame_paths) >= num_frames:
            break

    cap.release()
    return frame_paths


def compute_aesthetic_score(image_path: str) -> float:
    """Compute a visual aesthetic score using CLIP model."""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(
        text=["beautiful scenery", "aesthetic composition", "cinematic shot"],
        images=image,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
        score = outputs.logits_per_image.softmax(dim=1).mean().item()
    return round(score, 3)

# -------------------- API Endpoints --------------------

@app.get("/")
def root():
    return {
        "message": "Framesque API is running",
        "version": "1.1.0",
        "endpoints": {
            "upload_video": "POST /api/upload",
            "get_video_frames": "GET /api/video/{video_id}/frames",
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "videos_processed": len(videos)
    }

@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    latitude: float = 0.0,
    longitude: float = 0.0,
    user_id: str = "demo_user"
):
    """Upload a video, extract frames, and analyze them."""
    if not file.content_type.startswith('video/'):
        raise HTTPException(400, "File must be a video")

    # Save uploaded video
    video_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    videos[video_id] = {
        "id": video_id,
        "user_id": user_id,
        "filename": file.filename,
        "latitude": latitude,
        "longitude": longitude,
        "status": "processing",
        "uploaded_at": datetime.now().isoformat(),
    }

    # Extract frames
    frame_paths = extract_frames(save_path, FRAMES_DIR, num_frames=10)

    # Analyze frames
    analyzed_frames = []
    for i, path in enumerate(frame_paths):
        score = compute_aesthetic_score(path)
        frame_info = {
            "id": len(frames) + 1,
            "video_id": video_id,
            "frame_number": i,
            "frame_url": f"/frames/{os.path.basename(path)}",
            "aesthetic_score": score,
            "created_at": datetime.now().isoformat(),
        }
        frames[f"{video_id}_{i}"] = frame_info
        analyzed_frames.append(frame_info)

    # Save metadata
    videos[video_id]["status"] = "completed"
    videos[video_id]["frames_extracted"] = len(analyzed_frames)
    videos[video_id]["processed_at"] = datetime.now().isoformat()

    best_frame = max(analyzed_frames, key=lambda x: x["aesthetic_score"])

    return {
        "video_id": video_id,
        "status": "completed",
        "frames_extracted": len(analyzed_frames),
        "best_frame": best_frame,
        "message": "Video analyzed successfully"
    }


@app.get("/api/video/{video_id}/frames")
def get_video_frames(video_id: str):
    """Retrieve analyzed frames for a video."""
    if video_id not in videos:
        raise HTTPException(404, f"Video {video_id} not found")

    result = [f for f in frames.values() if f["video_id"] == video_id]
    result.sort(key=lambda x: x["aesthetic_score"], reverse=True)
    return {
        "video_id": video_id,
        "total_frames": len(result),
        "frames": result
    }

# -------------------- Run Locally --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
