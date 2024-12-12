import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# Load the trained SVM model
svm_model = joblib.load("svm_model.pkl")
print("Trained SVM model loaded successfully.")

# Define categories (same as used in training)
categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Path to the prediction folder
prediction_dir = r"C:\Users\gauta\Documents\Optimization\Final Project\seg_pred"

# Image dimensions (used for resizing)
IMG_SIZE = (150, 150, 3)

# Predict function
def predict_images(directory, model, categories):
    predictions = []  # List to store predictions
    image_paths = []  # List to store image paths

    for img_name in os.listdir(directory):
        try:
            # Read the image
            img_path = os.path.join(directory, img_name)
            img = imread(img_path)
            
            # Resize if necessary
            if img.shape != IMG_SIZE:
                img = resize(img, IMG_SIZE)

            # Flatten image for the SVM model
            img_flattened = img.flatten().reshape(1, -1)

            # Predict the class
            predicted_class = categories[model.predict(img_flattened)[0]]
            predictions.append(predicted_class)
            image_paths.append(img_path)

            print(f"Predicted: {predicted_class} for {img_name}")

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

    return image_paths, predictions

# Predict classes for all images in the prediction folder
image_paths, predicted_classes = predict_images(prediction_dir, svm_model, categories)

# Save predictions to a CSV file
output_data = pd.DataFrame({
    'Image Path': image_paths,
    'Predicted Class': predicted_classes
})
output_data.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")

# Optional: Visualize predictions
for img_path, pred_class in zip(image_paths, predicted_classes):
    img = imread(img_path)
    img = img / 255.0 if img.dtype == 'float64' else img.astype(np.uint8)  # Normalize for display
    plt.imshow(img)
    plt.title(f"Predicted Class: {pred_class}")
    plt.axis('off')
    plt.show()
