# Skin Disease Classification System

A web-based application that uses deep learning to classify common skin diseases from images. The system can identify 8 different types of skin conditions using a Convolutional Neural Network (CNN) model.

## Features

- **8 Disease Classes**: BA-cellulitis, BA-impetigo, FU-athlete-foot, FU-nail-fungus, FU-ringworm, PA-cutaneous-larva-migrans, VI-chickenpox, VI-shingles
- **Real-time Prediction**: Upload images and get instant predictions with confidence scores
- **Detailed Descriptions**: Each prediction includes a description of the condition
- **User-friendly Interface**: Modern, responsive web interface built with Bootstrap
- **Educational Tool**: Designed for educational purposes with proper disclaimers

## Dataset

The model was trained on a comprehensive dataset containing thousands of labeled images across 8 skin disease categories:
- Bacterial infections (Cellulitis, Impetigo)
- Fungal infections (Athlete's foot, Nail fungus, Ringworm)
- Parasitic infections (Cutaneous larva migrans)
- Viral infections (Chickenpox, Shingles)

## Model Architecture

- **Input**: 224x224 RGB images
- **Architecture**: CNN with multiple convolutional layers
- **Output**: Softmax classification for 8 disease classes
- **Training**: 10 epochs with Adam optimizer

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd skin-disease
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists**:
   Make sure `skindiseasemodel.h5` is in the project root directory.

## Usage

1. **Start the application**:
   ```bash
   python APP.PY
   ```

2. **Access the web interface**:
   Open your browser and go to `http://localhost:5000`

3. **Upload an image**:
   - Click "Choose File" to select a skin image
   - Click "Predict" to get the analysis
   - View the results with confidence score and description

## Project Structure

```
skin-disease/
├── APP.PY                 # Main Flask application
├── skindiseasemodel.h5    # Trained model file
├── skindisease.ipynb      # Jupyter notebook with model training
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/            # HTML templates
│   ├── index.html        # Main prediction interface
│   ├── about.html        # About page
│   ├── project.html      # Project details
│   └── contact.html      # Contact page
└── skin-disease-datasaet/ # Dataset directory
    └── images/           # Training images organized by class
```

## API Endpoints

- `GET /` - Main prediction interface
- `GET /about` - About page
- `GET /project` - Project details
- `GET /contact` - Contact page
- `POST /predict` - Image prediction endpoint
- `GET /api/classes` - Get all disease classes

## Technical Details

### Model Training
The model was trained using TensorFlow/Keras with the following specifications:
- **Input shape**: (224, 224, 3)
- **Convolutional layers**: 3 layers with max pooling
- **Dense layers**: 128 units with ReLU activation
- **Output layer**: 8 units with softmax activation
- **Loss function**: Sparse categorical crossentropy
- **Optimizer**: Adam

### Image Preprocessing
- Resize to 224x224 pixels
- Convert BGR to RGB
- Normalize pixel values (0-1)
- Add batch dimension

## Disclaimer

⚠️ **Important**: This application is designed for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for proper medical care.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- **Project Lead**: Ankur Saini
- **Email**: sainianku018@gmail.com

## Acknowledgments

- Dataset contributors
- TensorFlow/Keras community
- Medical professionals who provided guidance 
