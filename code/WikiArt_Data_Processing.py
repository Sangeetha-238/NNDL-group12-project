from PIL import Image
import os
import numpy as np
import argparse
import glob
import pandas as pd

parser = argparse.ArgumentParser(description="Preprocess images for GAN training")
parser.add_argument('--data_dir', type=str, required=True, help='Directory containing WikiArt dataset')
parser.add_argument('--target_size', type=int, nargs=2, default=(128, 128), help='Target size for images (width, height)')
parser.add_argument('--grayscale', action='store_true', help='Convert images to grayscale')
parser.add_argument('--csv_output_dir', type=str, required=True, help='Directory for storing CSV files with image paths and labels')

args = parser.parse_args()

def create_image_data_csv(root_dir, csv_file_path):
    data = []

    # Pattern to match all image files
    pattern = os.path.join(root_dir, '*', '*', '*.*')  
    
    for img_path in glob.glob(pattern):
        # Extract art movement and artist from the path
        parts = img_path.split(os.sep)
        art_movement, artist = parts[-3], parts[-2]
        data.append([img_path, art_movement, artist])

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data, columns=['Image_Path', 'Art_Movement', 'Artist'])
    df.to_csv(csv_file_path, index=False)

def preprocess_images(csv_path, target_size=(128, 128), grayscale=False):
    
    df_path = pd.read_csv(csv_path)
    
    for img_path in df_path['Image_Path']:
        
        try:
            # Open the image file
            with Image.open(img_path) as img:
                # Resize image
                img = img.resize(target_size).convert('RGB')

                # Convert to grayscale if needed
                if grayscale:
                    img = img.convert('L')

                # Convert to numpy array and normalize pixel values
                img_array = np.asarray(img)
                img_array = img_array / 255.0 if img_array.max() > 1 else img_array

                # Save the preprocessed image
                img = Image.fromarray((img_array * 255).astype(np.uint8))
                img.save(img_path)     

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    create_image_data_csv(args.data_dir, args.csv_output_dir)
    preprocess_images(args.csv_output_dir, target_size=tuple(args.target_size), grayscale=args.grayscale)
