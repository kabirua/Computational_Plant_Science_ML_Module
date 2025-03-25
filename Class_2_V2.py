'''
1. Revisit list
2. Simple code define matrix or 2D list
3. visualized it.
4. lableing too
5. create box and their labels
'''
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras import layers, models

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

# Create the base image with zeros (200x200)
a = np.zeros((200, 200))
many_lists = []
labels=[]
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

for i in range(0, 1000):
    a_copy = a.copy()  # Create a fresh copy of the base array
    a1 = boxes_square(a_copy)
    many_lists.append(a1)
    labels.append(1)

list_number=5
# Visualize 5 samples from the dataset
plot_samples(many_lists, labels, list_number)
