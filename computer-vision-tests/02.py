import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def histogram_equalization(image_array):
    # Compute the histogram
    histogram, bins = np.histogram(image_array.flatten(), bins=256, range=[0, 256])

    # Compute the cumulative distribution function (CDF)
    cdf = histogram.cumsum()

    # Normalize the CDF
    cdf_normalized = cdf * histogram.max() / cdf.max()

    # Perform histogram equalization
    equalized_image_array = np.interp(image_array.flatten(), bins[:-1], cdf_normalized)
    equalized_image_array = equalized_image_array.reshape(image_array.shape)

    # Convert the image array back to uint8
    equalized_image_array = equalized_image_array.astype(np.uint8)

    return equalized_image_array


def plot_grayscale_histogram(image_path):
    # Open the image and convert it to grayscale
    image = Image.open(image_path).convert('L')

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Perform histogram equalization
    equalized_image_array = histogram_equalization(image_array)

    # Plot the original grayscale histogram
    plt.subplot(1, 2, 1)
    plt.hist(image_array.flatten(), bins=256, color='gray')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Original Grayscale Histogram')

    # Plot the equalized grayscale histogram
    plt.subplot(1, 2, 2)
    plt.hist(equalized_image_array.flatten(), bins=256, color='gray')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Equalized Grayscale Histogram')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the histograms
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

    # Perform histogram equalization on each color channel
    equalized_red_channel = histogram_equalization(red_channel)
    equalized_green_channel = histogram_equalization(green_channel)
    equalized_blue_channel = histogram_equalization(blue_channel)

    # Plot the original RGB histograms
    plt.subplot(2, 2, 1)
    plt.hist(red_channel.flatten(), bins=256, color='red', alpha=0.5, label='Red')
    plt.hist(green_channel.flatten(), bins=256, color='green', alpha=0.5, label='Green')
    plt.hist(blue_channel.flatten(), bins=256, color='blue', alpha=0.5, label='Blue')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Original RGB Histogram')
    plt.legend()

    # Plot the equalized RGB histograms
    plt.subplot(2, 2, 2)
    plt.hist(equalized_red_channel.flatten(), bins=256, color='red', alpha=0.5, label='Red')
    plt.hist(equalized_green_channel.flatten(), bins=256, color='green', alpha=0.5, label='Green')
    plt.hist(equalized_blue_channel.flatten(), bins=256, color='blue', alpha=0.5, label='Blue')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Equalized RGB Histogram')
    plt.legend()

    # Display the histograms
    plt.tight_layout()
    plt.show()


plot_grayscale_histogram('./images/scarp.jpg')
plot_rgb_histogram('./images/scarp.jpg')