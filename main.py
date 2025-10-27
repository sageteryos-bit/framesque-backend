from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import random
import time
import uuid
from datetime import datetime

app = FastAPI(
    title="Framesque API",
    description="AI-powered background discovery for content creators",
    version="1.0.0"
)

# CORS - allow all origins for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (demo mode)
videos = {}
frames = {}

@app.get("/")
def root():
    return {
        "message": "Framesque API - Demo Mode",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "upload": "POST /api/upload",
            "video_status": "GET /api/video/{video_id}/status",
            "video_frames": "GET /api/video/{video_id}/frames"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "framesque-api",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    latitude: float = 40.7589,
    longitude: float = -73.9851,
    user_id: str = "demo_user"
):
    """Upload video and generate AI-scored frames (demo mode)"""
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(400, "File must be a video")
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    
    # Save video info
    videos[video_id] = {
        "id": video_id,
        "user_id": user_id,
        "filename": file.filename,
        "latitude": latitude,
        "longitude": longitude,
        "status": "completed",
        "frames_extracted": 10,
        "uploaded_at": datetime.now().isoformat(),
        "processed_at": datetime.now().isoformat()
    }
    
    # Generate 10 mock frames with realistic scores
    locations = [
        "Times Square, NYC",
        "Central Park, NYC",
        "Brooklyn Bridge, NYC",
        "Empire State Building, NYC",
        "Manhattan Skyline",
        "Hudson River Park",
        "SoHo District",
        "Greenwich Village",
        "Chelsea Market",
        "High Line Park"
    ]
    
    for i in range(10):
        frame_id = len(frames) + 1
        
        # Generate realistic aesthetic scores
        # Top frames score higher
        if i < 3:
            base_score = random.uniform(0.75, 0.95)
        elif i < 6:
            base_score = random.uniform(0.60, 0.80)
        else:
            base_score = random.uniform(0.45, 0.65)
        
        has_faces = random.choice([True, False]) if i > 5 else False
        
        frames[f"{video_id}_{i}"] = {
            "id": frame_id,
            "video_id": video_id,
            "frame_number": i,
            "frame_url": f"https://picsum.photos/seed/{video_id}_{i}/800/600",
            "aesthetic_score": round(base_score, 2),
            "has_faces": has_faces,
            "face_count": random.randint(1, 3) if has_faces else 0,
            "upvotes": random.randint(0, 50),
            "latitude": latitude + random.uniform(-0.01, 0.01),
            "longitude": longitude + random.uniform(-0.01, 0.01),
            "place_name": random.choice(locations),
            "place_type": random.choice(["landmark", "park", "street", "building"]),
            "created_at": datetime.now().isoformat()
        }
    
    return {
        "video_id": video_id,
        "status": "completed",
        "message": "Video processed successfully",
        "frames_extracted": 10
    }

@app.get("/api/video/{video_id}/status")
def get_video_status(video_id: str):
    """Get video processing status"""
    if video_id in videos:
        return videos[video_id]
    raise HTTPException(404, f"Video {video_id} not found")

@app.get("/api/video/{video_id}/frames")
def get_video_frames(
    video_id: str,
    min_score: float = 0.0,
    limit: int = 50
):
    """Get frames for a video, sorted by aesthetic score"""
    
    if video_id not in videos:
        raise HTTPException(404, f"Video {video_id} not found")
    
    result = []
    for frame_key, frame in frames.items():
        if frame["video_id"] == video_id and frame["aesthetic_score"] >= min_score:
            result.append(frame)
    
    # Sort by aesthetic score (highest first)
    result.sort(key=lambda x: x["aesthetic_score"], reverse=True)
    
    return result[:limit]

@app.post("/api/nearby")
def get_nearby_frames(
    latitude: float,
    longitude: float,
    radius_km: float = 5.0,
    min_score: float = 0.5,
    limit: int = 20
):
    """Get frames near a location"""
    result = []
    
    for frame in frames.values():
        if frame["aesthetic_score"] >= min_score:
            result.append(frame)
            if len(result) >= limit:
                break
    
    result.sort(key=lambda x: x["aesthetic_score"], reverse=True)
    return result

@app.post("/api/frame/{frame_id}/upvote")
def upvote_frame(frame_id: int, user_id: str = "demo_user"):
    """Upvote a frame"""
    
    for frame_key, frame in frames.items():
        if frame["id"] == frame_id:
            frame["upvotes"] += 1
            return {
                "frame_id": frame_id,
                "upvotes": frame["upvotes"],
                "message": "Upvote successful"
            }
    
    raise HTTPException(404, f"Frame {frame_id} not found")

@app.get("/api/place/{place_name}/top")
def get_top_frames_by_place(place_name: str, limit: int = 20):
    """Get top frames for a specific place"""
    result = []
    
    for frame in frames.values():
        if place_name.lower() in frame["place_name"].lower():
            result.append(frame)
    
    result.sort(key=lambda x: (x["upvotes"], x["aesthetic_score"]), reverse=True)
    return result[:limit]

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
