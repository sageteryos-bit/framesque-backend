from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import random
import uuid
from datetime import datetime
import os

# ✅ Ensure frames directory always exists (important for Render)
os.makedirs("frames", exist_ok=True)

app = FastAPI(
    title="Framesque API",
    description="AI-powered background discovery for content creators",
    version="1.0.1"
)

# Mount static files safely
app.mount("/frames", StaticFiles(directory="frames"), name="frames")

# Allow all CORS (for frontend demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores (temporary)
videos = {}
frames = {}

@app.get("/")
def root():
    return {
        "message": "Framesque API - Running",
        "status": "active",
        "version": "1.0.1",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "framesque-api",
        "version": "1.0.1",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    latitude: float = 40.7589,
    longitude: float = -73.9851,
    user_id: str = "demo_user"
):
    """Upload video and generate demo AI scores"""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video")

    video_id = str(uuid.uuid4())
    video_path = f"frames/{video_id}_{file.filename}"

    # Save uploaded video temporarily
    with open(video_path, "wb") as f:
        f.write(await file.read())

    videos[video_id] = {
        "id": video_id,
        "filename": file.filename,
        "status": "completed",
        "frames_extracted": 10,
        "uploaded_at": datetime.now().isoformat(),
    }

    # Generate mock AI frames
    locations = [
        "Times Square", "Central Park", "Brooklyn Bridge",
        "Empire State Building", "Manhattan Skyline",
        "Hudson River Park", "SoHo District", "Chelsea Market"
    ]

    for i in range(10):
        frame_id = len(frames) + 1
        frames[f"{video_id}_{i}"] = {
            "id": frame_id,
            "video_id": video_id,
            "frame_number": i,
            "frame_url": f"https://picsum.photos/seed/{video_id}_{i}/800/600",
            "aesthetic_score": round(random.uniform(0.5, 0.95), 2),
            "has_faces": random.choice([True, False]),
            "latitude": latitude + random.uniform(-0.01, 0.01),
            "longitude": longitude + random.uniform(-0.01, 0.01),
            "place_name": random.choice(locations),
            "created_at": datetime.now().isoformat(),
        }

    return {
        "video_id": video_id,
        "status": "completed",
        "message": "Video processed successfully",
        "frames_extracted": 10,
    }

@app.get("/api/video/{video_id}/frames")
def get_frames(video_id: str):
    if video_id not in videos:
        raise HTTPException(404, "Video not found")

    result = [
        f for f in frames.values() if f["video_id"] == video_id
    ]
    result.sort(key=lambda x: x["aesthetic_score"], reverse=True)
    return result

# Local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
