from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import os
import uuid

app = FastAPI()

# Ensure 'frames' directory exists
if not os.path.exists("frames"):
    os.makedirs("frames")

# Serve the frames folder publicly
app.mount("/frames", StaticFiles(directory="frames"), name="frames")

@app.get("/")
def home():
    return {"message": "Video background analyzer API is running!"}

@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_filename = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())

    # Open the video with OpenCV
    cap = cv2.VideoCapture(temp_filename)
    frame_urls = []
    count = 0

    if not cap.isOpened():
        return JSONResponse({"error": "Unable to open video file"}, status_code=400)

    # Extract every 30th frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
            frame_name = f"frame_{uuid.uuid4().hex[:8]}.jpg"
            frame_path = os.path.join("frames", frame_name)
            cv2.imwrite(frame_path, frame)
            # Build public URL
            url = f"https://framesque-backend.onrender.com/frames/{frame_name}"
            frame_urls.append(url)
            count += 1

    cap.release()
    os.remove(temp_filename)

    if not frame_urls:
        return JSONResponse({"message": "No frames extracted — try a longer video"}, status_code=400)

    return {"frames": frame_urls[:10]}  # limit to first 10
