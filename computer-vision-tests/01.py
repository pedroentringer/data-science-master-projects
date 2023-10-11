import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_grayscale_histogram(image_path):
    # Open the image and convert it to grayscale
    image = Image.open(image_path).convert('L')

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Plot the histogram
    plt.hist(image_array.flatten(), bins=256, color='gray')

    # Set the labels and title
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Grayscale Histogram')

    # Display the histogram
    plt.show()


def plot_rgb_histogram(image_path):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Separate the color channels
    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    blue_channel = image_array[:, :, 2]

    # Plot the histogram for the red channel
    plt.hist(red_channel.flatten(), bins=256, color='red', alpha=0.5, label='Red')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Red Channel Histogram')
    plt.legend()
    plt.show()

    # Plot the histogram for the green channel
    plt.hist(green_channel.flatten(), bins=256, color='green', alpha=0.5, label='Green')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Green Channel Histogram')
    plt.legend()
    plt.show()

    # Plot the histogram for the blue channel
    plt.hist(blue_channel.flatten(), bins=256, color='blue', alpha=0.5, label='Blue')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Blue Channel Histogram')
    plt.legend()
    plt.show()


#plot_grayscale_histogram('./images/stinkbug.png')
plot_rgb_histogram('./images/scarp.jpg')
