from PIL import Image
import numpy as np

# Read in image
image = Image.open('2.png')

# Truncate image to multiples of 8 in x and y
image = image.crop((0, 0, 8 * (image.width // 8), 8 * (image.height // 8)))

# Keep only one color component
image = image.convert('L')

# Convert intensities to integers between 0 and 255, and center around 0
image = np.array(image)
image = image.astype(np.int32) - 128

# Define matrix for DCT2 P
P = np.zeros((8, 8), dtype=np.float64)
for u in range(8):
    for v in range(8):
        if u == 0:
            P[u][v] = 1 / np.sqrt(8)
        else:
            P[u][v] = np.sqrt(2 / 8) * np.cos((2 * v + 1) * u * np.pi / 16)

def compress(image, P, Q, threshold=0):
    """Compress an image using DCT2 and quantization.

    Parameters:
    image (np.array): The image to be compressed.
    P (np.array): Matrix for DCT2.
    Q (np.array): Matrix for quantization.
    threshold (int): Threshold for filtering high frequencies.

    Returns:
    np.array: The compressed image.
    float: Compression rate.
    """
    # Initialize compressed image and number of non-zero coefficients
    compressed = np.empty_like(image)
    non_zero_coeffs = 0

    # Iterate over 8x8 blocks of the image
    for i in range(0, image.shape[0], 8):
        for j in range(0, image.shape[1], 8):
            # Get current block
            block = image[i:i+8, j:j+8]

            # Apply DCT2
            D = np.dot(P, np.dot(block, P.T))

            # Apply quantization
            D = np.round(D / Q)

            # Filter high frequencies and set low frequencies to 0
            D[threshold:, :] = 0
            D[:, threshold:] = 0

            # Update number of non-zero coefficients
            non_zero_coeffs += np.count_nonzero(D)

            # Add transformed block to compressed image
            compressed[i:i+8, j:j+8] = D

    # Calculate compression rate
    rate = non_zero_coeffs / (image.shape[0] * image.shape[1])

    return compressed, rate

Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])

compressed_image, compression_rate = compress(image, P, Q, threshold=4)

def decompress(compressed, P, Q):
    """Decompress an image using the inverse DCT2 and dequantization.

    Parameters:
    compressed (np.array): The compressed image.
    P (np.array): Matrix for DCT2.
    Q (np.array): Matrix for quantization.

    Returns:
    np.array: The decompressed image.
    """
    # Initialize decompressed image
    decompressed = np.empty_like(compressed)

    # Iterate over 8x8 blocks of the compressed image
    for i in range(0, compressed.shape[0], 8):
        for j in range(0, compressed.shape[1], 8):
            # Get current block
            block = compressed[i:i+8, j:j+8]

            # Dequantize
            D = block * Q

            # Apply inverse DCT2
            M = np.dot(P.T, np.dot(D, P))

            # Add transformed block to decompressed image
            decompressed[i:i+8, j:j+8] = M

    return decompressed
decompressed_image = decompress(compressed_image, P, Q)

def compare_images(original, decompressed):
    """Compare two images using the L2 relative norm.

    Parameters:
    original (np.array): The original image.
    decompressed (np.array): The decompressed image.

    Returns:
    float: The L2 relative error between the two images.
    """
    # Calculate L2 relative error
    error = np.linalg.norm(original - decompressed) / np.linalg.norm(original)

    return error

error = compare_images(image, decompressed_image)

# Convert image back to range between 0 and 255
decompressed_image = decompressed_image + 128
decompressed_image = decompressed_image.astype(np.uint8)

# Save image
Image.fromarray(decompressed_image).save('decompressed.png')
