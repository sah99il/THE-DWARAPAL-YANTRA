import cv2
import numpy as np
import os
import glob

def simulate_id_degradation(image):
    """
    Applies a degradation pipeline to simulate a low-quality ID scan.
    
    Args:
        image (numpy.ndarray): The input image (should be 112x112).
        
    Returns:
        numpy.ndarray: The degraded image.
    """
    # 1. Downsampling/Upsampling for pixelation
    downsampled = cv2.resize(image, (50, 50), interpolation=cv2.INTER_NEAREST)
    pixelated_img = cv2.resize(downsampled, (112, 112), interpolation=cv2.INTER_NEAREST)
    
    # 2. Gaussian Blur to soften details
    blurred_img = cv2.GaussianBlur(pixelated_img, (5, 5), 0)
    
    # 3. Salt-and-Pepper Noise for grain
    noisy_img = blurred_img.copy()
    noise_density = 0.02
    
    # Salt
    num_salt = np.ceil(noise_density * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_img[coords[0], coords[1], :] = 255
    
    # Pepper
    num_pepper = np.ceil(noise_density * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_img[coords[0], coords[1], :] = 0
    
    return noisy_img

def main():
    """
    Main execution function to process images from data/raw and save to data/attacked.
    """
    raw_dir = 'data/raw/'
    attacked_dir = 'data/attacked/'
    
    # Ensure the output directory exists
    os.makedirs(attacked_dir, exist_ok=True)
    
    # Find all .jpg and .png images
    image_paths = glob.glob(os.path.join(raw_dir, '*.jpg')) + glob.glob(os.path.join(raw_dir, '*.png'))
    
    if not image_paths:
        print(f"No .jpg or .png images found in '{raw_dir}'.")
        print("Please add some images to process (e.g., by running 'utils/check_dataset.py' if available).")
        return
        
    print(f"Found {len(image_paths)} images to process.")
    
    for i, image_path in enumerate(image_paths):
        # Read the image
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
            
        # Ensure image is 112x112 before degradation
        img_resized = cv2.resize(img, (112, 112))
        
        # Apply the degradation pipeline
        degraded_img = simulate_id_degradation(img_resized)
        
        # Save the processed image
        base_filename = os.path.basename(image_path)
        output_filename = f"grainy_{base_filename}"
        output_path = os.path.join(attacked_dir, output_filename)
        
        cv2.imwrite(output_path, degraded_img)
        print(f"({i+1}/{len(image_paths)}) Saved degraded image to: {output_path}")
        
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
