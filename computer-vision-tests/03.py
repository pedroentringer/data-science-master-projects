import matplotlib.pyplot as plt
from PIL import Image


def upsample_image(image_path, scale_factor):
    # Open the image
    image = Image.open(image_path)

    # Calculate the new size
    width, height = image.size
    new_width = width * scale_factor
    new_height = height * scale_factor

    # Perform up-sampling using the nearest neighbor algorithm
    upsampled_image = image.resize((new_width, new_height), resample=Image.NEAREST)

    return upsampled_image


# Example usage
image_path = './images/scarp.jpg'
scale_factor = 2  # Upsample by a factor of 2

upsampled_image = upsample_image(image_path, scale_factor)

# Save the upsampled image
# upsampled_image.save('upsampled_image.jpg')

# Display the original and upsampled images using matplotlib
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# Original image
axs[0].imshow(Image.open(image_path))
axs[0].set_title('Original Image')
axs[0].axis('off')

# Upsampled image
axs[1].imshow(upsampled_image)
axs[1].set_title('Upsampled Image')
axs[1].axis('off')

plt.show()