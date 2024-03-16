import os
import cv2
import numpy as np

def remove_background(image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Use adaptive thresholding to separate foreground from background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask to extract the foreground
    mask = np.zeros_like(image)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    
    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result

def process_images(input_directory, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over each subfolder (test and val) in the input directory
    for subfolder in os.listdir(input_directory):
        subfolder_path = os.path.join(input_directory, subfolder)
        if os.path.isdir(subfolder_path):
            # Create corresponding subfolders in the output directory
            output_subfolder_path = os.path.join(output_directory, subfolder)
            os.makedirs(output_subfolder_path, exist_ok=True)

            # Iterate over each image in the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    # Read the image
                    image_path = os.path.join(subfolder_path, filename)
                    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    
                    # Apply background removal
                    output_image = remove_background(input_image)
                    
                    # Save the background removed image
                    output_path = os.path.join(output_subfolder_path, filename)
                    cv2.imwrite(output_path, output_image)

    print("Background removal complete.")

# Path to the directory containing the test and val subfolders
input_directory = 'dataset\SignImage48x48'

# Output directory to save background removed images
output_directory = 'dataset\datasetconverted'

# Process images in the test and val subfolders
process_images(input_directory, output_directory)
