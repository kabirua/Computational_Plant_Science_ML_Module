import numpy as np
import joblib
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, models

def plot_samples(X, y, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        axes[i].imshow(X[i], cmap='gray')
        axes[i].set_title(f"Label: {y[i]}")
        axes[i].axis('off')
    plt.show()

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
        if img is not None:
            img = cv2.resize(img, (200, 200))  # Resize for consistency
            images.append(img)
            labels.append(label)
    return images, labels

# Load images from both folders
healthy_images, healthy_labels = load_images_from_folder("C:/Users/hossaink/Desktop/Computational Plant Sciences/5th_class/OneDrive_2025-03-04/Alstonia Scholaris (P2)/healthy", label=0)
diseased_images, diseased_labels = load_images_from_folder("C:/Users/hossaink/Desktop/Computational Plant Sciences/5th_class/OneDrive_2025-03-04/Alstonia Scholaris (P2)/diseased", label=1)

# Combine datasets
X = np.array(healthy_images + diseased_images)
y = np.array(healthy_labels + diseased_labels)

# Normalize pixel values (scale between 0 and 1)
X = X / 255.0

# Reshape to match model input format (assuming CNN expects 4D input)
X = X.reshape(-1, 200, 200, 1)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plot_samples(X_train,y_train, num_samples=5)

# Define a simple CNN model
model = models.Sequential([
    layers.Input(shape=(200, 200, 1)),  # Specify input shape
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification: 1 for rectangle, 0 for square
])


# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))



# Save model as pickle
joblib.dump(model, "leafclassify.pkl")
print("Model saved successfully!")

# Load model and make predictions
loaded_model = joblib.load("leafclassify.pkl")
y_pred = loaded_model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")  # Convert probabilities to class labels

# Evaluate model
conf_matrix = confusion_matrix(y_test, y_pred_classes)
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Test Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

plot_samples(X_test,y_pred_classes, num_samples=5)

