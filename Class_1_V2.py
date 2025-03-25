'''
1. Revisit list
2. Simple code define matrix or 2D list
3. visualized it.
Task to do: make it functions
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
    fig, axes = plt.subplots(1,num_samples, figsize=(15, 5))
    # for i in range(num_samples):
    axes.imshow(X, cmap='gray')
    # axes[i].set_title(f"Label: {y[i]}")
    axes.axis('off')
    plt.show()

# Create the base image with zeros (200x200)
a = np.zeros((200, 200))
num_samples = 1

def random(height,width):
    # Ensure height ≠ width for rectangles
    rect_height = np.random.randint(20, height // 2)
    rect_width = np.random.randint(20, width // 2)
    return rect_height,rect_width

def rectangle(a1,):
     # Extract dimensions from the passed matrix

    # Ensure height ≠ width for rectangles
    rect_height = np.random.randint(20, height // 2)
    rect_width = np.random.randint(20, width // 2)
    # Random position for the rectangle
    x = np.random.randint(0, height - rect_height)
    y = np.random.randint(0, width - rect_width)

    a1[x:x + rect_height, y:y + rect_width] = 1
    return a1
height, width = a.shape
rect_height,rect_width= random(height,width)
a1=rectangle(a,height,width)

# Visualize 5 samples from the dataset
plot_samples(a1, list_number=1)


