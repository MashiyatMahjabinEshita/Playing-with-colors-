!pip install pillow numpy matplotlib scikit-learn  # installing all necessary libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from google.colab import drive     #calling out drive for help

drive.mount('/content/drive')  # mounting google drive

def get_palette(image, num_colors=5):  # Function for generating a color palette of my favourite image
    image = image.resize((100, 100))  #compacting image for processing
    image_np = np.array(image)


    if len(image_np.shape) == 3 and image_np.shape[2] == 3:      # Checking if the image has 3 color channels (RGB)
        pixels = image_np.reshape(-1, 3)
    else:
        # Handle grayscale or RGBA images
        # Convert to RGB if necessary
        image = image.convert('RGB')
        image_np = np.array(image)
        pixels = image_np.reshape(-1, 3)

    # Use KMeans to find dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Get the colors
    colors = kmeans.cluster_centers_.astype(int)

    # Create a palette visualization
    palette = np.zeros((100, 100 * num_colors, 3), dtype=int)
    for i, color in enumerate(colors):
        palette[:, i * 100:(i + 1) * 100] = color

    # Plot the palette
    plt.imshow(palette)
    plt.axis('off')
    plt.show()

    return colors

# Specify the path to your image in Google Drive
image_path = '/content/drive/My Drive/pic.png'  # Update this path

# Process the image
image = Image.open(image_path)
num_colors = 5  # Set the number of colors in the palette
colors = get_palette(image, num_colors)
print("Dominant Colors (RGB):", colors)
