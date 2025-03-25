'''
1. Revisit list
2. Simple code define matrix or 2D list
3. visualized it.
4. lableing too
5. create box and their labels
6. train with CNN

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

'''
Functions 
'''
def plot_samples(X, y, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        axes[i].imshow(X[i], cmap='gray')
        axes[i].set_title(f"Label: {y[i]}")
        axes[i].axis('off')
    plt.show()



def boxes_rectangle(a1):

    height, width = a1.shape  # Extract dimensions from the passed matrix
    while True:
        rect_height = np.random.randint(20, height // 2)
        rect_width = np.random.randint(20, width // 2)
        if rect_height != rect_width:
            break

    # Ensure height ≠ width for rectangles
    rect_height = np.random.randint(20, height // 2)
    rect_width = np.random.randint(20, width // 2)
    # Random position for the rectangle
    x = np.random.randint(0, height - rect_height)
    y = np.random.randint(0, width - rect_width)

    a1[x:x + rect_height, y:y + rect_width] = 1

    return a1

# Task for student, they will do similar function for square
def boxes_square(a1):
    height, width = a1.shape  # Extract dimensions from the passed matrix

    # Generate square size
    square_size = np.random.randint(20, height // 2)

    # Random position for the square
    x = np.random.randint(0, height - square_size)
    y = np.random.randint(0, width - square_size)


    a1[x:x + square_size, y:y + square_size] = 1

    return a1

# Create the base image with zeros (200x200)
a = np.zeros((200, 200))
many_lists1 = []
labels1=[]

many_lists2 = []
labels2=[]

for i in range(0, 1000):
    a_copy = a.copy()  # Create a fresh copy of the base array
    a1 = boxes_rectangle(a_copy)
    many_lists1.append(a1)
    labels1.append(0)

for i in range(0, 1000):
    a_copy = a.copy()  # Create a fresh copy of the base array
    a1 = boxes_square(a_copy)
    # a1[:, :] = 1
    many_lists2.append(a1)
    labels2.append(1)


# Convert to numpy arrays for easier handling
X = np.array(many_lists1 + many_lists2)  # Merged images, (1000,200,200)
y = np.array(labels1 + labels2)         # Merged labels
# Reshape to add the channel dimension (needed for CNN)
X = X.reshape(-1, 200, 200, 1)  # Shape: (1000, 200, 200, 1), -1 add new dimention at the end
print(X.shape)

# Visualize 5 samples from the dataset
plot_samples(X, y, 5)


# Split data into train and test sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # every time split will be the same
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
print("Trained size:",X_train.shape)
# Ensure data is in float32 format (required by TensorFlow)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Visualize 20 samples from the dataset
plot_samples(X_train, y_train, num_samples=10)

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

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=2, batch_size=32) # 1600/32 = 50 batches, each 32 sample
# history = model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_test, y_test))

# Step 1: Predict using the model
y_pred = model.predict(X_test)
# Since it's binary classification, convert probabilities to class labels (0 or 1)
y_pred_classes = (y_pred > 0.5).astype("int32")  # Threshold of 0.5 for binary classification

# Step 2: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)


# Step 3: Calculate Accuracy, Precision, Recall
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)


print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Test Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")









