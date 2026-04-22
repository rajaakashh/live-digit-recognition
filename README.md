Live Handwritten Digit Recognition (Computer Vision)

A real-time computer vision project that recognizes handwritten digits using a webcam. The system captures live input, processes the image, and predicts the digit using a trained ONNX model.


## Overview

This project demonstrates a complete end-to-end pipeline for handwritten digit recognition. It uses OpenCV for image processing and an ONNX model for fast and efficient inference.

The user can draw a digit and webcame capture it and the system predicts it in real time.

---

## Tech Stack

- Python  
- OpenCV  
- NumPy  
- ONNX Runtime  
- Pre-trained MNIST model  

---

## Features

- Real-time webcam-based digit recognition  
- Region of Interest (ROI) for drawing digits  
- Image preprocessing (grayscale, blur, thresholding)  
- Contour-based digit extraction  
- MNIST-style normalization (28x28 input)  
- ONNX model inference for fast predictions  
- Displays predicted digit with confidence score  

---

##  How It Works

1. Capture live video from webcam  
2. Define a region of interest (ROI)  
3. Convert ROI to grayscale  
4. Apply blur and thresholding  
5. Extract the digit using contours  
6. Resize and normalize to 28x28 format  
7. Pass it to ONNX model  
8. Display predicted digit on screen  

---
## 📷 Demo

![Live Demo](assets/demo.png)
##  How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

