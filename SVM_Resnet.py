import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.utils import to_categorical
import cv2
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Define dataset paths
base_dir = r"C:\Users\gauta\Documents\Optimization\Final Project"  
train_dir = os.path.join(base_dir, "seg_train")
test_dir = os.path.join(base_dir, "seg_test")

# Define categories
categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# ResNet requires 224x224 images
RESNET_IMG_SIZE = (224, 224)

# Load ResNet Model (Feature Extractor)
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Preprocessing function
def preprocess_image(img):
    img = cv2.resize(img, RESNET_IMG_SIZE)  # Resize image to 224x224
    img = preprocess_input(img)  # Normalize as per ResNet
    return img

# Load images and preprocess
def load_data(data_dir):
    images = []
    labels = []
    for category in categories:
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = preprocess_image(img)
                images.append(img)
                labels.append(category)
    return np.array(images), np.array(labels)

print("Loading training data...")
X_train, y_train = load_data(train_dir)
print("Loading testing data...")
X_test, y_test = load_data(test_dir)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Extract features using ResNet
print("Extracting features using ResNet...")
X_train_resnet = resnet_model.predict(X_train, verbose=1)
X_test_resnet = resnet_model.predict(X_test, verbose=1)

# Train SVM
print("Training SVM...")
svm = SVC(kernel='linear', C=1, probability=True)
svm.fit(X_train_resnet, y_train_encoded)

# Save the SVM model
svm_model_path = os.path.join(base_dir, "svm_resnet_model.pkl")
joblib.dump(svm, svm_model_path)
print(f"SVM model saved to {svm_model_path}")

# Evaluate SVM
print("Evaluating SVM...")
y_pred = svm.predict(X_test_resnet)
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=categories))
