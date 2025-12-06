from fastapi.testclient import TestClient
from main import app
import cv2
import numpy as np
import io

client = TestClient(app)

def test_index():
    response = client.get("/")
    assert response.status_code == 200

def test_classify_frame():
    # Create a dummy image (red square)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:] = (0, 0, 255) # Red
    
    # Encode to jpg
    _, encoded_img = cv2.imencode('.jpg', img)
    
    # Create file-like object
    file_obj = io.BytesIO(encoded_img.tobytes())
    
    response = client.post(
        "/classify_frame",
        files={"file": ("test.jpg", file_obj, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    print("Classify Response:", data)
    assert data["status"] == "success"
    assert "top_result" in data
    assert "predictions" in data
    # We expect some prediction, likely "shirt" or "bag" or something random for a red square, 
    # but as long as it returns a valid CLIP prediction, it's working.

def test_learn_object():
    # Create a dummy image
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:] = (255, 0, 0) # Blue
    
    _, encoded_img = cv2.imencode('.jpg', img)
    file_obj = io.BytesIO(encoded_img.tobytes())
    
    response = client.post(
        "/learn",
        data={"label": "Blue Square"},
        files={"file": ("blue.jpg", file_obj, "image/jpeg")}
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"

if __name__ == "__main__":
    test_index()
    test_classify_frame()
    test_learn_object()
    print("All tests passed!")
