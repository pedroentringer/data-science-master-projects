from PIL import Image


def upsample_image(image_path, scale_factor, interpolation):
    # Open the image
    image = Image.open(image_path)

    # Calculate the new size
    width, height = image.size
    new_width = width * scale_factor
    new_height = height * scale_factor

    # Perform up-sampling using the specified interpolation method
    upsampled_image = image.resize((new_width, new_height), resample=interpolation)

    return upsampled_image


# Example usage
image_path = './images/stinkbug.png'
scale_factor = 2  # Upsample by a factor of 2

# Upsample using nearest neighbor interpolation
upsampled_nearest = upsample_image(image_path, scale_factor, Image.NEAREST)

# Upsample using bilinear interpolation
upsampled_bilinear = upsample_image(image_path, scale_factor, Image.BILINEAR)

# Upsample using bicubic interpolation
upsampled_bicubic = upsample_image(image_path, scale_factor, Image.BICUBIC)

# Save the upsampled images
upsampled_nearest.save('./images/upsampled_nearest.png')
upsampled_bilinear.save('./images/upsampled_bilinear.png')
upsampled_bicubic.save('./images/upsampled_bicubic.png')
