# Image Classification using Convolutional Neural Networks (CNNs)

This repository contains code for a simple image classification model using TensorFlow and Convolutional Neural Networks (CNNs).The project aims to classify images into two classes: 'Happy' and 'Sad'. The code utilizes TensorFlow and OpenCV for image processing and CNN model creation.

## Requirements
- Python 3.x
- TensorFlow
- OpenCV (cv2)
- Matplotlib


## Usage

**1. Data Preparation:**
- Organize your image dataset into a folder structure where images for each class are stored in separate directories inside a parent directory named 'data'. 
- Supported image formats are: JPEG, JPG, BMP, PNG.

**2. Data Preprocessing:**
- The script checks for valid image formats and removes files that do not match the specified formats.

**3. Model Training:**
- The CNN model is built using TensorFlow's Keras API.
- The model architecture consists of Convolutional layers, MaxPooling, Flatten, and Dense layers.
- Training is performed on the prepared dataset with a split of 70% training, 20% validation, and 10% testing data.

**4. Model Evaluation:**
- After training, the model's performance metrics (Precision, Recall, Accuracy) are evaluated on the test dataset.

**5. Prediction:**
- A sample image ('happytest.jpg') is provided for inference.
- The trained model predicts the class ('Happy' or 'Sad') of the sample image.

**6. Model Saving and Loading:**
- The trained model is saved as 'imageclassifier.h5' in the 'models' directory.
- Loading the saved model is demonstrated, followed by using it for inference on the sample image.


## Files
- image_classification_cnn.py: Main Python script containing the complete workflow.
- happytest.jpg: Sample image for prediction.
- models/: Directory to store the saved model.
- logs/: Directory for TensorBoard logs (used for visualization during training).
## Instructions
1.	Ensure the required libraries are installed.
2.	Organize your dataset following the provided directory structure.
3.	Run the image_classification_cnn.py script to train the model and perform inference.


