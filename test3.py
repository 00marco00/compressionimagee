from PIL import Image
import numpy as np

# Read the image
im = Image.open('1.jpg')

# Truncate the image to multiples of 8 in x and y
im = im.crop((0, 0, im.size[0] // 8 * 8, im.size[1] // 8 * 8))

# Transform intensities to integers between 0 and 255, then center to be between -128 and 127
im_data = np.array(im, dtype=np.int8)
im_data = im_data - 128

# Define the matrix for the DCT2 transformation P
P = np.zeros((8, 8), dtype=np.float64)
for k in range(8):
    for n in range(8):
        if k == 0:
            P[k, n] = 1 / np.sqrt(8)
        else:
            P[k, n] = np.sqrt(2 / 8) * np.cos((np.pi / 8) * (n + 0.5) * k)

# Compression

# Define the quantization matrix Q
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float64)

# Initialize the compressed image data matrix
compressed_im_data = np.empty(im_data.shape, dtype=np.int8)

# Initialize the number of non-zero coefficients
non_zero_coefficients = 0

# Iterate over 8x8 blocks of the image
for i in range(0, im_data.shape[0], 8):
    for j in range(0, im_data.shape[1], 8):
        # Iterate over color channels
        for c in range(im_data.shape[2]):
            # Get the current block
            block = im_data[i:i+8, j:j+8, c]
            
            # Apply the change of basis D = PMP^T
            D = np.dot(P, np.dot(block, P.T))
            
            # Apply the quantization matrix Q element-wise
            D = np.round(D / Q)
            
            # Filter high frequencies and set coefficients on the last rows and columns to 0
            D[4:, :] = 0
            D[:, 4:] = 0
            
            # Store the new block in the compressed image data matrix
            compressed_im_data[i:i+8, j:j+8, c] = D
            
            # Count the number of non-zero coefficients
            non_zero_coefficients += np.count_nonzero(D)

# Calculate the compression rate
compression_rate = non_zero_coefficients / (im_data.shape[0] * im_data.shape[1] * im_data.shape[2])

# Decompression

# Initialize the decompressed image data matrix
decompressed_im_data = np.empty(im_data.shape, dtype=np.int8)

# Iterate over 8x8 blocks of the image
for i in range(0, im_data.shape[0], 8):
    for j in range(0, im_data.shape[1], 8):
        # Iterate over color channels
        for c in range(im_data.shape[2]):
            # Get the current block
            block = compressed_im_data[i:i+8, j:j+8, c]
            
            # Multiply by the quantization matrix Q element-wise
            D = block * Q
            
            # Apply the inverse transformation P^T D P
            block = np.dot(P.T, np.dot(D, P))
            
            # Store the new block in the decompressed image data matrix
            decompressed_im_data[i:i+8, j:j+8, c] = block


# Post-processing

# Transform values between -128 and 127 to be between 0 and 1
decompressed_im_data = decompressed_im_data / 255

# Calculate the relative L2 norm between the original and decompressed image data
l2_norm = np.linalg.norm(im_data - decompressed_im_data) / np.linalg.norm(im_data)

# Save the decompressed image
Image.fromarray(np.uint8(decompressed_im_data * 255)).save('decompressed_image.jpg')
