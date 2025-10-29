# framesque-backend

AI-powered photo spot finder backend

## What the analyzer returns
- **Frame metrics** – normalized brightness, contrast, saturation, clutter, and colorfulness for every extracted frame.
- **Backdrop scoring** – per-frame scores optimized separately for video intros and still photography so you can pick the right scene.
- **Pose coaching** – lighting and styling tips plus MediaPipe-driven posture feedback for the strongest pose in each context.

## Local setup
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000` and will expose frame images from the `frames/` directory.

## Try the `/analyze/` endpoint
1. Grab a short MP4 clip such as `sample.mp4`.
2. Send it to the API:
   ```bash
   curl -X POST "http://127.0.0.1:8000/analyze/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample.mp4"
   ```
3. Review the JSON response:
   - `analysis` lists each sampled frame with metrics and the calculated video/photo scores.
   - `best_backdrops.video` and `.photography` identify the top frames for each use case.
   - `pose_guidance` provides lighting tweaks, styling ideas, and posture coaching drawn from MediaPipe landmarks.

## Quick sync & redeploy checklist
1. Pull the latest default-branch changes to keep your repo in sync:
   ```bash
   git pull origin main
   ```
2. Commit and push the updated analyzer back to GitHub:
   ```bash
   git add .
   git commit -m "Update analyzer"
   git push origin main
   ```
3. In Render, open the **framesque-api** service, choose **Manual Deploy → Deploy latest commit**, and wait for the build (Python 3.12.4 via `render.yaml`) to finish before testing the health check.
