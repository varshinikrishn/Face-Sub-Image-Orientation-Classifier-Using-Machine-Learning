# Face-Sub-Image-Orientation-Classifier-Using-Machine-Learning
A machine learning pipeline to classify the orientation of facial sub-images (30x30, 50x50, 90x90 pixels) from the "Labeled Faces in the Wild" dataset.   The system generates rotated sub-images from full face images, trains separate scikit-learn classifiers for each sub-image size, and evaluates performance against a standardized test set. 

Key focus:
- Sub-image preprocessing and augmentation
- PCA dimensionality reduction
- Supervised classification using KNN and other scikit-learn classifiers
- Hyperparameter tuning and evaluation

This repository supports reproducible experiments for orientation detection tasks in computer vision and image analytics.

## Repository Structure
- `train.py` – Training script for models
- `model.30.joblib`, `model.50.joblib`, `model.90.joblib` – Trained classifiers
- `report.pdf` – Technical report with system design, experiments, and results
- `eval1.joblib` – Evaluation dataset (download separately)
