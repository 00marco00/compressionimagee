from PIL import Image
import math

# Ouvrir l'image
image = Image.open("2.png")

# Afficher les informations sur l'image
print(image.size)
print(image.mode)

# Afficher l'image
image.show()

# Calculer la largeur et la hauteur de l'image
width, height = image.size

'''
# Tronquer l'image à des multiples de 8 en x et y
cropped_image = image.crop((0, 0, width - width % 8, height - height % 8))

# Afficher l'image tronquée
cropped_image.show()
'''

# Calculate the new size of the image
new_width = math.floor(width / 8) * 8
new_height = math.floor(height / 8) * 8

# Crop the image to the new size
image = image.crop((0, 0, new_width, new_height))

#converti l'image en niveau de gris
image = image.convert("L")

# Afficher l'image
image.show()





