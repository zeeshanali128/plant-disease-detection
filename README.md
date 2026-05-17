# Plant Disease Detection System

A deep learning-based system for detecting and classifying plant diseases using Convolutional Neural Networks (CNN).

## Project Structure

```
plant-disease-detection/
├── dataset/              # Training, validation, and test data
│   ├── train/
│   ├── val/
│   └── test/
├── model/                # Trained model files
├── src/                  # Source code modules
│   ├── data_preparation.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── predict.py
├── app/                  # Flask web application
│   ├── app.py
│   ├── templates/        # HTML templates
│   └── static/           # CSS, JS, images
├── notebooks/            # Jupyter notebooks for exploration
├── outputs/              # Results, plots, visualizations
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Features

- **Disease Detection**: Classify plant diseases using deep learning
- **Model Interpretability**: Grad-CAM visualizations for explainability
- **Web Interface**: Flask-based REST API and web application
- **Performance Evaluation**: Comprehensive metrics and confusion matrices

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
```python
python src/train.py
```

### Making Predictions
```python
python src/predict.py --image path/to/image.jpg
```

### Running Web App
```bash
python app/app.py
```

## Models

The project uses a CNN architecture with:
- Multiple convolutional layers
- Max pooling for dimensionality reduction
- Dropout for regularization
- Softmax activation for multi-class classification

## Results

Results are saved in the `outputs/` directory including:
- Training/validation plots
- Confusion matrices
- Grad-CAM heatmaps

## License

MIT License
