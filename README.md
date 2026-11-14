# Face Attribute Prediction using Machine Learning

A computer vision and machine learning project that predicts facial attributes — **Age, Gender, and Race** — from facial images using Histogram of Oriented Gradients (HOG) for feature extraction and LightGBM for model training.

The project also supports real-time facial attribute prediction using a webcam.

## Features

- Facial image preprocessing using OpenCV.
- HOG-based feature extraction.
- Age prediction using LightGBM Regressor.
- Gender classification using LightGBM Classifier.
- Race classification using LightGBM Classifier.
- Real-time webcam input for facial attribute prediction.
- Model serialization and loading using Joblib.

## Tech Stack

- Python
- OpenCV
- NumPy
- Pandas
- Scikit-image
- Scikit-learn
- LightGBM
- Joblib

## Project Structure

```text
Face-Attribute-Prediction-ML/
│
├── Face_recognition.py
├── model.py
├── Models/
│   ├── model_age.pkl
│   ├── model_gd.pkl
│   └── model_race.pkl
│
├── Data/
│   └── crop_part1/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How It Works

### 1. Image Preprocessing
- Images are resized to 200 × 200 pixels.
- Images are converted into grayscale.

### 2. Feature Extraction
HOG (Histogram of Oriented Gradients) is used to extract image features representing facial structures and gradient information.

### 3. Model Training
Three separate LightGBM models are trained:
- Age: Regression.
- Gender: Classification.
- Race: Classification.

The dataset is split into training and testing sets using an 80:20 ratio.

### 4. Prediction
The trained models are loaded using Joblib and used to predict facial attributes from extracted image features.

The project supports webcam input through OpenCV.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Face-Attribute-Prediction-ML.git
cd Face-Attribute-Prediction-ML
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Train the models:

```bash
python model.py
```

Run real-time webcam prediction:

```bash
python Face_recognition.py
```

Press `q` to exit the webcam window.

## Model Evaluation

The project evaluates:
- Gender classification using Accuracy.
- Race classification using Accuracy.
- Age regression using Mean Absolute Error (MAE).

Actual evaluation results can be added here after training and testing.

## Limitations

- Predictions depend on the quality and diversity of the training dataset.
- Model performance may vary across different lighting conditions, camera qualities, and demographic groups.
- The current implementation does not include face detection or multi-face tracking.
- Predictions are estimates and should not be treated as reliable assessments of a person's attributes.

## Future Improvements

- Integrate face detection for automatic face localization.
- Improve model performance through hyperparameter tuning.
- Add a graphical user interface.
- Evaluate performance across different datasets and demographic groups.
- Optimize real-time inference.

## Disclaimer

This project is developed for educational and research purposes. Predictions involving age, gender, and race can be inaccurate and may reflect biases present in the training dataset. They should not be used for consequential decisions.

## Author

**Anshul Pareek**

B.Tech – Electrical Engineering (Rail Engineering)  
Gati Shakti Vishwavidyalaya

- GitHub: https://github.com/AnshulPareek
- LinkedIn: https://linkedin.com/in/anshulpareek54
