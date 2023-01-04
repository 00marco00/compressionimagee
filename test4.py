import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image

# Lire l'image
im = Image.open("2.png")

# Tronquer l'image à des multiples de 8 en x et y
im = im.crop((0, 0, im.width - im.width % 8, im.height - im.height % 8))

# Ne garder qu'une composante (couleur) pour avoir une matrice 2D
im_matrix = np.array(im.convert("L"))

# Transformer les intensités en entiers entre 0 et 255, puis centrer pour se ramener entre −128 et 127
im_matrix = (im_matrix - 128).astype(np.int8)

# Définir la matrice de passage en fréquentiel de la DCT2 P
P = np.empty((8, 8))
for i in range(8):
    for j in range(8):
        P[i][j] = np.cos(np.pi * (2 * i + 1) * j / 16)

# Pour chaque bloc 8 x 8 de l'image :
compressed_image = np.empty_like(im_matrix)
for i in range(0, im_matrix.shape[0], 8):
    for j in range(0, im_matrix.shape[1], 8):
        # Appliquer le changement de base D = PMP^T
        D = np.dot(np.dot(P, im_matrix[i:i+8, j:j+8]), P.T)
        # Appliquer la matrice de quantification terme à terme D./Q et prendre la partie entière
        Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])
        D_quant = (D / Q).astype(np.int32)
        # et/ou Filtrer les hautes fréquences et mettre à 0 les cœfficients sur les dernières lignes et colonnes dans la matrice
        # ...
        # Stocker ces nouveaux blocs 8 x 8 dans la matrice compressée/débruitée
        compressed_image[i:i+8, j:j+8] = D_quant

# Compter le nombre de cœfficients non nuls pour obtenir le# Pour chaque bloc 8 x 8 :
decompressed_image = np.empty_like(compressed_image)
for i in range(0, compressed_image.shape[0], 8):
    for j in range(0, compressed_image.shape[1], 8):
        # Multiplier par la matrice Q terme à terme
        D_dequant = compressed_image[i:i+8, j:j+8] * Q
        # Appliquer la transformée inverse P^T D P
        im_block = np.dot(np.dot(P.T, D_dequant), P)
        # Réassembler la matrice décompressée/débruitée
        decompressed_image[i:i+8, j:j+8] = im_block

# Post-processing :
# Comparer cette matrice avec la matrice originale (en norme L2 relative par exemple)
relative_error = np.linalg.norm(im_matrix - decompressed_image) / np.linalg.norm(im_matrix)
print(f"Erreur relative: {relative_error:.2f}")

# Le faire pour chaque canal de couleur si nécessaire
# ...

# Re-transformer les
# Re-transformer les valeurs entre −128 et 127 en réels entre 0 et 1 et sauver l'image (pour la comparer visuellement)
decompressed_image = (decompressed_image + 128) / 255
decompressed_image = Image.fromarray((decompressed_image * 255).astype(np.uint8))
decompressed_image.save("decompressed_image.png")