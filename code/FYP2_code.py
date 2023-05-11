
# First year project 2
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries, slic, find_boundaries
from skimage import io, morphology, transform
from PIL import Image


# Function to get the height of a lesion
def get_height(lesion_mask):
    pixels_in_col = np.sum(lesion_mask, axis = 0)
    
    pixels1 = pixels_in_col > 0
    pixels1 = pixels1.astype(np.int8)
    
    height = np.max(pixels_in_col)
    
    return height

# Function to get the width of a lesion
def get_width(lesion_mask):
    pixels_in_row = np.sum(lesion_mask, axis = 1)
    
    pixels1 = pixels_in_row > 0
    pixels1 = pixels1.astype(np.int8)
    
    width = np.max(pixels_in_row)
    
    return width

# Function to get the diameter of a lesion
def get_diameter(lesion_mask):
    diameter = 0
    
    for _ in range(7):
        pixels_in_row = np.sum(lesion_mask, axis = 0)
        
        pixels1 = pixels_in_row > 0
        pixels1 = pixels1.astype(np.int8)
        width = np.max(pixels_in_row)
        
        if width > diameter:
            diameter = width
            
        lesion_mask = transform.rotate(lesion_mask, 45)
        
        lesion_mask = lesion_mask > 0
        lesion_mask = lesion_mask.astype(np.int8)
        
    return diameter

# Function to get the perimeter of a lesion
def get_perimeter(lesion_mask):
    smaller_mask = morphology.disk(3)
    mask_eroded = morphology.binary_erosion(lesion_mask, smaller_mask)
    perimeter_img = lesion_mask - mask_eroded
    perimeter_pixels = np.sum(perimeter_img)
    return perimeter_img, perimeter_pixels

# Functions that gets and prepares images to be worked with
def get_image(lesion_image_path, lesion_mask_path):
    image = Image.open(lesion_image_path)
    image = image.convert("RGB")
     
    img_mask = io.imread(lesion_mask_path)
    img_mask = img_mask > 0
    img_mask = img_mask.astype(np.int8)
    
    return image, img_mask



def main():
    # Get the lesion image and its mask
    image, img_mask = get_image("images/PAT_1109_437_254.png", "images/PAT_1109_437_254_MASK.png")

    # Segmentation of images by color using slic algorythm
    segments = slic(image, mask = img_mask, n_segments = 10, start_label = 1, convert2lab = True, enforce_connectivity = False)
    plt.imshow(segments)
    plt.show()

    # Height of lesion
    print("Height is", get_height(img_mask))

    # Width of lesion
    print("Width is", get_width(img_mask))

    # Diameter of a lesion
    print("Diameter is", get_diameter(img_mask))

    # Get perimeter of the lesion
    perimeter_img, perimeter_pixels = get_perimeter(img_mask)
    print("Perimeter is", perimeter_pixels)
    plt.imshow(perimeter_img, cmap = "gray")
    plt.show()
    
    # Boundaries related (might need)
    boundaries = find_boundaries(segments, connectivity = 10)
    plt.imshow(boundaries)
    plt.show()
    boundaries_mask = mark_boundaries(img_mask, boundaries, color = (1, 1, 0))
    plt.imshow(boundaries_mask)
    plt.show()
    
    image.close()

if __name__ == "__main__":
    main()