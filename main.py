import cv2
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import threading
import time
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load CLIP Model (OpenAI)
# We use 'openai/clip-vit-base-patch32' for a good balance of speed and accuracy
from transformers import CLIPProcessor, CLIPModel
import torch

print("Loading CLIP model...")
try:
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    print("CLIP Model loaded.")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    model = None
    processor = None

# Optional: Keep YOLO for object detection (bounding boxes) if needed, 
# but for Visual Search/Classification, CLIP is primary.
# We will skip loading YOLO to save memory unless explicitly requested.
yolo_model = None 
# try:
#     yolo_model = YOLO('yolo11l.pt')
# except:
#     pass

# Global storage for learned objects
# Format: {"label": str, "embedding": numpy_array}
learned_objects = []

# Global video source
camera_lock = threading.Lock()

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0) # Default to webcam for scanner
        if not self.video.isOpened():
            print("Error: Could not open webcam.")
    
    def __del__(self):
        if self.video is not None:
            self.video.release()
    
    def get_frame(self):
        if self.video is None or not self.video.isOpened():
             # Try to reopen
            self.video = cv2.VideoCapture(0)
            if not self.video.isOpened():
                return None

        success, image = self.video.read()
        if not success:
            return None
        
        # We can add overlay here if needed, but for now we keep it simple
        # as the main use case is client-side scanning.
        
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

def get_embedding(image):
    # Use CLIP to extract feature vector
    try:
        if model and processor:
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            # Normalize
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features[0].cpu().numpy()
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

@app.post("/learn")
async def learn_object(label: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"status": "error", "message": "Invalid image"}

    # Crop to center for learning
    h, w, _ = image.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped_image = image[start_y:start_y+min_dim, start_x:start_x+min_dim]

    # Get embedding
    embedding = get_embedding(cropped_image)
    if embedding is None:
        return {"status": "error", "message": "Could not extract features. Model might not support embedding."}

    # Store
    learned_objects.append({"label": label, "embedding": embedding})
    print(f"Learned new object: {label}")
    
    return {"status": "success", "message": f"Learned '{label}'"}

@app.post("/classify_frame")
async def classify_frame(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"status": "error", "message": "Invalid image"}

    h, w, _ = image.shape
    center_x = w // 2
    center_y = h // 2

    # Run YOLOv11 Object Detection - REMOVED (Using CLIP)
    # results = model(image, verbose=False) if model else []
    results = []
    
    top_result = "Unknown"
    predictions = []
    
    # 1. Check Learned Objects (Few-Shot/Custom)
    best_match_label = None
    best_match_score = 0.0
    
    # Crop for embedding check (Center Crop)
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped_for_embed = image[start_y:start_y+min_dim, start_x:start_x+min_dim]

    if learned_objects:
        current_embedding = get_embedding(cropped_for_embed)
        if current_embedding is not None:
            for obj in learned_objects:
                sim = cosine_similarity(current_embedding, obj["embedding"])
                if sim > best_match_score:
                    best_match_score = sim
                    best_match_label = obj["label"]
    
    if best_match_label and best_match_score > 0.85:
        top_result = f"{best_match_label.upper()} {int(best_match_score*100)}%"
        predictions.append({"name": best_match_label, "score": int(best_match_score*100)})
        predictions.append({"name": "(Custom Match)", "score": 0})
        
    else:
        # 2. Fallback to CLIP Zero-Shot Classification
        if model and processor:
            # Define a broad list of categories for "Visual Search"
            # This can be expanded or loaded from a file
            candidate_labels = [
                "person", "man", "woman", "face",
                "cat", "dog", "bird", "animal",
                "car", "bicycle", "motorcycle", "bus", "truck",
                "chair", "table", "couch", "bed", "tv", "laptop", "mouse", "keyboard", "cell phone",
                "bottle", "cup", "fork", "knife", "spoon", "bowl",
                "banana", "apple", "orange", "sandwich", "pizza", "donut", "cake",
                "potted plant", "book", "clock", "vase", "scissors", "teddy bear", "toothbrush",
                "shoes", "bag", "backpack", "hat", "glasses", "watch", "jewelry",
                "shirt", "pants", "dress", "jacket", "coat"
            ]
            
            try:
                inputs = processor(text=candidate_labels, images=cropped_for_embed, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                logits_per_image = outputs.logits_per_image # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1) # we can use softmax to get probabilities
                
                # Get top 3
                top_probs, top_indices = probs.topk(3)
                
                top_probs = top_probs[0].cpu().numpy()
                top_indices = top_indices[0].cpu().numpy()
                
                top_result = f"{candidate_labels[top_indices[0]].upper()} {int(top_probs[0]*100)}%"
                
                for i in range(len(top_indices)):
                    label = candidate_labels[top_indices[i]]
                    score = int(top_probs[i] * 100)
                    predictions.append({"name": label, "score": score})
                    
            except Exception as e:
                print(f"CLIP classification error: {e}")
                top_result = "Error"
        else:
             top_result = "Model not loaded"

    return {
        "status": "success",
        "top_result": top_result,
        "predictions": predictions
    }

if __name__ == '__main__':
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
