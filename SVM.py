import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Define dataset paths
train_dir = r"C:\Users\gauta\Documents\Optimization\Final Project\seg_train"
test_dir = r"C:\Users\gauta\Documents\Optimization\Final Project\seg_test"

# Define categories
categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Image dimensions (skip resizing if all images are already uniform)
IMG_SIZE = (150, 150, 3)

# Function to load and preprocess data
def load_data(directory, categories):
    flat_data = []
    target = []
    for category in categories:
        print(f"Loading category: {category}")
        path = os.path.join(directory, category)
        for img in os.listdir(path):
            try:
                img_array = imread(os.path.join(path, img))  # Read image
                if img_array.shape != IMG_SIZE:  # Check shape
                    img_array = resize(img_array, IMG_SIZE)  # Resize to (150, 150, 3)
                flat_data.append(img_array.flatten())  # Flatten image
                target.append(categories.index(category))  # Add label as index
            except Exception as e:
                print(f"Error reading image {img}: {e}")
    return np.array(flat_data), np.array(target)

# Load training data
X_train, y_train = load_data(train_dir, categories)
print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")

# Load testing data
X_test, y_test = load_data(test_dir, categories)
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

# Train the SVM model
svm_model = SVC(kernel='linear', C=1.0)  # Linear kernel
svm_model.fit(X_train, y_train)

# Predict on the test data
y_pred = svm_model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize predictions
for i in range(5):  # Display 5 random test samples
    img = X_test[i].reshape(150, 150, 3)  # Reshape to original image dimensions
    plt.imshow(img)
    plt.title(f"True: {categories[y_test[i]]}, Predicted: {categories[y_pred[i]]}")
    plt.axis('off')
    plt.show()

import joblib

# Save the trained SVM model
joblib.dump(svm_model, "svm_model.pkl")
print("Trained SVM model saved as svm_model.pkl")