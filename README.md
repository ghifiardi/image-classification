# Visual Search with CLIP 🔍

A powerful visual search and object classification application using **OpenAI's CLIP model**. This app allows you to classify objects in real-time using a webcam and "learn" new objects for custom identification.

## Features
- **Zero-Shot Classification**: Recognizes common objects (person, cat, laptop, etc.) without training.
- **Custom Object Learning**: "Teach" the app to recognize specific items (e.g., "My Blue Pen") using few-shot learning.
- **Mobile Access**: Easily access the app from your phone using the built-in secure tunnel.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/ghifiardi/image-classification.git
    cd image-classification
    ```

2.  Create a virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

### Run Locally
Start the server:
```bash
python main.py
```
Open [http://localhost:8001](http://localhost:8001) in your browser.

### Run on Mobile 📱
To access the app from your mobile phone (requires internet):

1.  Start the app in one terminal:
    ```bash
    python main.py
    ```
2.  In a second terminal, run the share script:
    ```bash
    ./share.sh
    ```
3.  Scan the **QR Code** that appears in your terminal with your phone.

## Tech Stack
- **FastAPI**: Web server.
- **CLIP (Transformers)**: Vision-Language model for embeddings and classification.
- **OpenCV**: Image processing.
- **Pinggy**: Secure tunneling for mobile access.
