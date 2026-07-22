MNIST Digit Classifier — CNN + Streamlit

A hand-drawn digit recognizer built with a custom Convolutional Neural Network trained on the MNIST dataset, wrapped in an interactive Streamlit web app. Draw a digit on a canvas or upload an image, and the model predicts the number in real time with a confidence score and full probability breakdown.

How it works
Model: A CNN with three convolutional blocks (32 → 64 → 128 channels), each followed by ReLU activation and max pooling, then an adaptive average pooling layer and two fully connected layers producing 10-class digit predictions (0–9).
Training: The model (Main.ipynb) is trained from scratch on the standard MNIST dataset (60,000 training images, 10,000 test images) using the Adam optimizer and cross-entropy loss over 50 epochs, reaching 99.35% test accuracy. The trained weights are saved to mnist_cnn.pth.
Inference app: The Streamlit app (app.py) loads the trained model and offers two ways to get a prediction:
Draw: Sketch a digit directly on an in-browser canvas (via streamlit-drawable-canvas).
Upload: Upload a PNG/JPG image of a handwritten digit.
Both paths preprocess the image (grayscale conversion, resizing to 28×28, inversion, and normalization to match the MNIST training pipeline) before feeding it to the model. Results include the predicted digit, a confidence percentage, and a bar chart of the full softmax probability distribution across all 10 classes, plus a download button for the processed image.
Tech stack

Python, PyTorch, Torchvision, Streamlit, streamlit-drawable-canvas, NumPy, Pillow.