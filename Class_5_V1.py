import numpy as np
import cv2 # pip install opencv-python
import os # OS module provide function that interact with operating system, file, path, directory 

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder): # list of all files present in the given folder 
        img_path = os.path.join(folder, filename) # join the file name with folder directory
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
        if img is not None:
            img = cv2.resize(img, (200, 200))  # Resize for consistency
            images.append(img)
            labels.append(label)
    return images, labels

# Load images from both folders \Alstonia Scholaris (P2)
healthy_images, healthy_labels = load_images_from_folder("C:/Users/hossaink/Desktop/Computational Plant Sciences/5th_class/OneDrive_2025-03-04/Alstonia Scholaris (P2)/healthy", label=0)
diseased_images, diseased_labels = load_images_from_folder("C:/Users/hossaink/Desktop/Computational Plant Sciences/5th_class/OneDrive_2025-03-04/Alstonia Scholaris (P2)/diseased", label=1)

# Combine datasets
X = np.array(healthy_images + diseased_images)
y = np.array(healthy_labels + diseased_labels)

# Normalize pixel values (scale between 0 and 1)
X = X / 255.0
print(X)

