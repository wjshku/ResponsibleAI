# Grad-CAM Web Application

A Flask-based web application for real-time Grad-CAM visualization of car scratch detection models.

## Features

- **Interactive Model Loading**: Load different models and datasets through the web interface
- **Real-time Visualization**: Generate Grad-CAM overlays instantly in the browser
- **Random Sampling**: Get random samples from the dataset for analysis
- **Responsive Design**: Clean, modern web interface that works on different screen sizes

## Installation

1. Install dependencies:
```bash
cd interpret_cnn/gradcam_app
pip install -r requirements.txt
```

2. Make sure the parent directory structure includes:
   - `detection_car/` (with data_loader.py, models.py)
   - `interpret_cnn/` (with utils.py)

## Usage

1. Start the Flask application:
```bash
cd interpret_cnn/gradcam_app
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the web interface to:
   - Load a model and dataset
   - Generate random Grad-CAM visualizations
   - Specify particular sample indices

## API Endpoints

- `POST /load_model`: Load model and dataset
- `POST /get_sample`: Get Grad-CAM visualization for a sample

## Configuration

The web app uses the same model loading and preprocessing logic as `gradcam_analyze.py`:

- **Model Loading**: Supports any PyTorch model saved with the same format
- **Data Preprocessing**: Uses the same evaluation transforms as the training pipeline
- **Grad-CAM**: Uses the same Grad-CAM calculation from `utils.py`
- **Visualization**: Creates the same 2x2 grid layout (original, original+Grad-CAM, processed, processed+Grad-CAM)

## Default Settings

- **Model Path**: `../detection_car/models/model_20251006_172724/cnn_best.pth`
- **Data Directory**: CarDD-TE dataset from GenAI_Results
- **Target Size**: Automatically detected from model metadata
- **Grad-CAM Threshold**: 10% (top 10% activated areas)

## Troubleshooting

- **Model Loading Issues**: Ensure the model path is correct and the model file exists
- **Data Loading Issues**: Check that the data directory contains the expected structure
- **Memory Issues**: For large datasets, consider using a smaller sample size or running on a machine with more RAM
- **Port Issues**: If port 5000 is occupied, modify the port in `app.py`