#!/bin/bash

# Exit on any error
set -e

echo "Updating system packages..."
sudo apt-get update

echo "Installing required system libraries..."
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup completed successfully!"
