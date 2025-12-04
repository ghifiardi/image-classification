#!/bin/bash
# Activate virtual environment (reuse the one from surveillance-system if compatible, or create new)
# We will reuse the existing venv in ../surveillance-system/venv for simplicity as requirements are similar
# But to be safe, let's just assume we run this from the visual-search dir and use the parent venv

source ../surveillance-system/venv/bin/activate

# Install requirements if needed (fastapi, uvicorn, ultralytics already installed)
# pip install -r requirements.txt

# Run the app
python main.py
