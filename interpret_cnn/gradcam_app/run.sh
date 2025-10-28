#!/bin/bash
# Run the Grad-CAM web application

echo "Starting Grad-CAM Web Application..."
echo "Make sure you have activated the conda environment: conda activate genai"
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "genai" ]]; then
    echo "Warning: genai conda environment not detected."
    echo "Please run: conda activate genai"
    echo ""
fi

# Start the Flask application
python app.py