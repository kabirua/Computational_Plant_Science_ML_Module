'''
1. Revisit list
2. Simple code define matrix or 2D list
3. visualized it.
'''
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras import layers, models

'''
Functions 
'''
def plot_samples(X,list_number=1):
    fig, axes = plt.subplots(1,list_number, figsize=(15, 5))
    # for i in range(num_samples):
    axes.imshow(X, cmap='gray')
    # axes[i].set_title(f"Label: {y[i]}")
    axes.axis('off')
    plt.show()

# Create the base image with zeros (200x200)
a = np.zeros((200, 200))


height, width = a.shape  # Extract dimensions from the list

# Random size for the rectangle
# This generates a random integer for the rectangle's height
rect_height = np.random.randint(20, height // 2) # '//' floor division
# This generates a random integer for the rectangle's width
rect_width = np.random.randint(20, width // 2)

# Random position for the rectangle
# This generates a random integer x, which represents the starting vertical position (y-axis) of the rectangle.
x = np.random.randint(0, height - rect_height) # x is height - rect_height, ensuring the rectangle fits inside the array vartically.
# this generates a random integer y for the starting horizontal position (x-axis) of the rectangle.
y = np.random.randint(0, width - rect_width)

a[x:x + rect_height, y:y + rect_width] = 1


# Visualize 5 samples from the dataset
plot_samples(a, list_number=1)


