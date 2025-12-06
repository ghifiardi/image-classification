#!/bin/bash
echo "----------------------------------------------------------------"
echo "Ensure your app is running in another terminal (python main.py)"
echo "----------------------------------------------------------------"
echo "Starting secure tunnel to localhost:8001..."
echo "Scan the QR code below with your mobile phone:"
echo ""
# Use Pinggy with QR code enabled
ssh -p 443 -R0:localhost:8001 qr@a.pinggy.io
