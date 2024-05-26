from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def load_image(path):
    """Load an image from a file path."""
    return imread(path)

def save_image(image, path):
    """Save an image to a specified file path."""
    imsave(path, image)
    

def rgb_to_grayscale(image):
    """Convert an RGB image to grayscale."""
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

def normalize_image(image):
    """Normalize image to range 0-1."""
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def show_images(images, titles=None, cmap='gray', figsize=(15, 5)):
    """Display a list of images with optional titles."""
    n = len(images)
    titles = titles or [''] * n
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def apply_filter(image, kernel):
    """Apply a convolutional filter to an image."""
    return convolve(image, kernel)