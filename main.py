import cv2
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import threading
import time

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load YOLOv11 Classification model
# This will automatically download 'yolo11n-cls.pt' if not present
print("Loading YOLOv11 Classification model...")
model = YOLO('yolo11n-cls.pt')
print("Model loaded.")

# Global video source
camera_lock = threading.Lock()
current_top_result = "Waiting..."

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0) # Default to webcam for scanner
        if not self.video.isOpened():
            print("Error: Could not open webcam.")
    
    def __del__(self):
        if self.video is not None:
            self.video.release()
    
    def get_frame(self):
        global current_top_result
        if self.video is None or not self.video.isOpened():
             # Try to reopen
            self.video = cv2.VideoCapture(0)
            if not self.video.isOpened():
                return None

        success, image = self.video.read()
        if not success:
            return None
        
        # Run YOLOv11 Classification
        # predict() returns a list of Results objects
        results = model(image, verbose=False)
        
        # Get top-5 predictions
        # The 'probs' attribute contains probabilities
        if results and results[0].probs:
            top5_indices = results[0].probs.top5
            top5_conf = results[0].probs.top5conf
            
            # Update global variable for UI overlay (optional, but good for debugging)
            # We will draw the top result on the frame
            
            names = results[0].names
            
            # Draw top result prominently
            top_idx = top5_indices[0]
            top_name = names[top_idx]
            top_score = top5_conf[0].item()
            
            current_top_result = f"{top_name} ({top_score:.2f})"
            
            # Draw UI on frame (Mobile Scanner Style)
            height, width, _ = image.shape
            
            # 1. Draw a "Scanner Box" in the center
            box_size = 300
            x1 = (width - box_size) // 2
            y1 = (height - box_size) // 2
            x2 = x1 + box_size
            y2 = y1 + box_size
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 2. Draw corners for cool effect
            corner_len = 20
            # Top-Left
            cv2.line(image, (x1, y1), (x1 + corner_len, y1), (0, 255, 0), 4)
            cv2.line(image, (x1, y1), (x1, y1 + corner_len), (0, 255, 0), 4)
            # Top-Right
            cv2.line(image, (x2, y1), (x2 - corner_len, y1), (0, 255, 0), 4)
            cv2.line(image, (x2, y1), (x2, y1 + corner_len), (0, 255, 0), 4)
            # Bottom-Left
            cv2.line(image, (x1, y2), (x1 + corner_len, y2), (0, 255, 0), 4)
            cv2.line(image, (x1, y2), (x1, y2 - corner_len), (0, 255, 0), 4)
            # Bottom-Right
            cv2.line(image, (x2, y2), (x2 - corner_len, y2), (0, 255, 0), 4)
            cv2.line(image, (x2, y2), (x2, y2 - corner_len), (0, 255, 0), 4)
            
            # 3. Draw Top Prediction Label
            label = f"{top_name.upper()} {int(top_score*100)}%"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            # Background for text
            cv2.rectangle(image, (x1, y1 - 30), (x1 + w + 10, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # 4. List other predictions at bottom
            y_offset = y2 + 30
            for i in range(1, 3): # Show 2nd and 3rd
                if i < len(top5_indices):
                    idx = top5_indices[i]
                    name = names[idx]
                    score = top5_conf[i].item()
                    text = f"{i+1}. {name}: {int(score*100)}%"
                    cv2.putText(image, text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    y_offset += 25

        # Encode the frame in JPEG format
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

camera = VideoCamera()

def gen(camera):
    while True:
        with camera_lock:
            frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            cv2.waitKey(100)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen(camera), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) # Use port 8001 to avoid conflict with surveillance
