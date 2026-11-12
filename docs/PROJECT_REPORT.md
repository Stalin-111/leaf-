# DeepLeaf AI — Project Report Starter

## Abstract
DeepLeaf AI is an end-to-end computer vision application that uses a Convolutional Neural Network to classify plant leaf images into healthy or disease categories. The system combines model training, quantitative evaluation, a Flask inference API, SQLite history, and a responsive web interface.

## Objectives
1. Build a reproducible image classification pipeline.
2. Train a custom CNN for plant disease recognition.
3. Evaluate using multiple classification metrics.
4. Expose inference through a REST API.
5. Provide a usable browser interface and prediction history.

## Methodology
Images are resized to 224x224 RGB, normalized, and augmented during training. The CNN uses stacked convolution, batch normalization, max-pooling, global average pooling, dropout, and softmax classification. Early stopping and learning-rate reduction are used to reduce overfitting.

## Limitations
Curated datasets can be easier than real field conditions. Lighting, background, camera quality, disease severity, and multiple simultaneous diseases can reduce performance. Results should therefore be reported on held-out and external test images.

## Future work
Transfer learning with MobileNet/EfficientNet, Grad-CAM visualization, field-image evaluation, multilingual guidance, mobile deployment, and model monitoring.
