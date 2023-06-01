# First year project 2
import os
import colorthief
import csv
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
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

# Function that gets the area of a lesion
def get_area(image_mask):
    return np.sum(image_mask)

# Function that gets the compactness of a lesion
def get_compactness(perimeter, area):
    return round((perimeter**2) / (4 * np.pi * area), 3)

# Function that gets the symmetry of a lesion
def get_symmetry(image_mask):
    mask_flip = cv2.flip(image_mask, 0)
    symmetry_image = mask_flip - image_mask
    symmetry_image = symmetry_image != 0
    symmetry_image = symmetry_image.astype(np.int8)
    symmetry_pixels = np.sum(symmetry_image) / 2
    return symmetry_image, symmetry_pixels

def crop_lesion(image_path, mask_path):  
    image = plt.imread(image_path)
    mask = plt.imread(mask_path)
    image = image[:,:,:3]

    image[mask == 0] = 0
    plt.imshow(image)     
    return image

def get_features(image_path, image_mask_path):
    image_mask = io.imread(image_mask_path)
    image_mask = image_mask > 0
    image_mask = image_mask.astype(np.int8)
    
    colors_cancerous = []
    current_directory = os.getcwd()
    filename = os.path.basename(image_path)
    
    crop = crop_lesion(image_path, image_mask_path)
    plt.imsave(f"{current_directory}/{filename[:-4]}_CROP.png", crop)
    color_thief = colorthief.ColorThief(f"{current_directory}/{filename[:-4]}_CROP.png")
    colors_cancerous = color_thief.get_palette(color_count = 2)
    
    height = get_height(image_mask)
    width = get_width(image_mask)
    diameter = get_diameter(image_mask)
    
    area = get_area(image_mask)
    perimeter_image, perimeter_pixel = get_perimeter(image_mask)
    compactness = get_compactness(perimeter_pixel, area)
    
    symmetry_image , symmetry_pixel = get_symmetry(image_mask)
    return width, height, diameter, perimeter_pixel, area, compactness, symmetry_pixel, colors_cancerous
    

def main():
     
    counter1 = 0
    counter2 = 0
    
    filenames = []
    filenames_masks = []
    images = []
    for filename in glob.glob('images/Cancerous_Lesions/*.png'):
        filenames.append(filename[25:])
        image = Image.open(filename)
        image = image.convert("RGB")
        images.append(image)
        counter1 += 1
              
    image_masks = []
    for filename in glob.glob('images/Cancerous_Masks/*.png'):
        image_mask = io.imread(filename)
        filenames_masks.append(filename[23:])
        image_mask = image_mask > 0
        image_mask = image_mask.astype(np.int8)
        image_masks.append(image_mask)
             
    for filename in glob.glob('images/Non_Cancerous_Lesions/*.png'):
        image = Image.open(filename)
        filenames.append(filename[29:])
        image = image.convert("RGB")
        images.append(image)
        counter2 += 1
            
    for filename in glob.glob('images/Non_Cancerous_Masks/*.png'):
        image_mask = io.imread(filename)
        filenames_masks.append(filename[27:])
        image_mask = image_mask > 0
        image_mask = image_mask.astype(np.int8)
        image_masks.append(image_mask)
         
    counter = counter1 + counter2
    areas = []
    heights = []
    widths = []
    diameters = []
    perimeter_pixels = []
    perimeter_images = []
    compactnesses = []
    symmetry_images = []
    symmetry_pixels = []
    colors_cancerous = []
    colors_non_cancerous = []
    current_directory = os.getcwd()
    for i in range(counter):
        if i < counter1:
            crop = crop_lesion(f"images/Cancerous_Lesions/{filenames[i]}", f"images/Cancerous_Masks/{filenames_masks[i]}")
            plt.imsave(f"{current_directory}/images/Cropped_Cancerous_Lesions/{filenames[i][:-4]}_CROP.png", crop)
            color_thief = colorthief.ColorThief(f"images/Cropped_Cancerous_Lesions/{filenames[i][:-4]}_CROP.png")
            color_cancerous = color_thief.get_palette(color_count = 2)
            colors_cancerous.append(color_cancerous)
        else:
            crop = crop_lesion(f"images/Non_Cancerous_Lesions/{filenames[i]}", f"images/Non_Cancerous_Masks/{filenames_masks[i]}")
            plt.imsave(f"{current_directory}/images/Cropped_Non_Cancerous_Lesions/{filenames[i][:-4]}_CROP.png", crop)
            color_thief = colorthief.ColorThief(f"images/Cropped_Non_Cancerous_Lesions/{filenames[i][:-4]}_CROP.png")
            color_non_cancerous = color_thief.get_palette(color_count = 2)
            colors_non_cancerous.append(color_non_cancerous)
        
        heights.append(get_height(image_masks[i]))
        widths.append(get_width(image_masks[i]))
        diameters.append(get_diameter(image_masks[i]))
        
        area = get_area(image_masks[i])
        areas.append(area)
        
        perimeter_image, perimeter_pixel = get_perimeter(image_masks[i])
        perimeter_pixels.append(perimeter_pixel)
        perimeter_images.append(perimeter_image)
        
        compactnesses.append(get_compactness(perimeter_pixel, area))
        
        symmetry_image , symmetry_pixel = get_symmetry(image_masks[i])
        symmetry_images.append(symmetry_image)
        symmetry_pixels.append(symmetry_pixel)
        
    last_i = 0    
    with open("features.csv", "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(["Width", "Height", "Diameter", "Perimeter", "Area", "Symmetry", "Compactness", "R1", "G1", "B1", "R2", "G2", "B2", "R3", "G3", "B3", "Cancerous"])
        for i in range(counter1):
            writer.writerow([widths[i], heights[i], diameters[i], perimeter_pixels[i], areas[i], symmetry_pixels[i], compactnesses[i], colors_cancerous[i][0][0], colors_cancerous[i][0][1], colors_cancerous[i][0][2], colors_cancerous[i][1][0], colors_cancerous[i][1][1], colors_cancerous[i][1][2], colors_cancerous[i][2][0], colors_cancerous[i][2][1], colors_cancerous[i][2][2], "True"])
            last_i = i
        last_i += 1
        for i in range(counter2):
            writer.writerow([widths[last_i], heights[last_i], diameters[last_i], perimeter_pixels[last_i], areas[last_i], symmetry_pixels[last_i], compactnesses[last_i], colors_non_cancerous[i][0][0], colors_non_cancerous[i][0][1], colors_non_cancerous[i][0][2], colors_non_cancerous[i][1][0], colors_non_cancerous[i][1][1], colors_non_cancerous[i][1][2], colors_non_cancerous[i][2][0], colors_non_cancerous[i][2][1], colors_non_cancerous[i][2][2],"False"])
            last_i += 1
    print("done")
            

if __name__ == "__main__":
    main()