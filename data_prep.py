import os
import cv2
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Configurations
DATASET_PATH = 'Driver Drowsiness Dataset (DDD)'
PROCESSED_PATH = 'dataset_processed'
IMG_SIZE = 112

def prepare_directories():
    if os.path.exists(PROCESSED_PATH):
        shutil.rmtree(PROCESSED_PATH)
    
    for split in ['train', 'validation', 'test']:
        for cls in ['open', 'closed']:
            os.makedirs(os.path.join(PROCESSED_PATH, split, cls), exist_ok=True)

def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    # Resize to 112x112
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Normalize
    img = img.astype('float32') / 255.0
    return img

def split_and_save_data():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Directory '{DATASET_PATH}' not found. Please place data here.")
        return False
        
    for src_cls, dst_cls in [('Non Drowsy', 'open'), ('Drowsy', 'closed')]:
        cls_path = os.path.join(DATASET_PATH, src_cls)
        if not os.path.exists(cls_path):
             print(f"Warning: Directory '{cls_path}' not found. Ensure dataset structure is correct.")
             continue
             
        # Support deep folder scanning if images are nested (e.g. within Drowsy / open eyes subdirs if any exist)
        images = []
        for root, dirs, files in os.walk(cls_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    images.append(os.path.join(root, f))
        
        # Split: 70% Train, 15% Val, 15% Test
        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
        
        splits = {'train': train_imgs, 'validation': val_imgs, 'test': test_imgs}
        
        for split_name, img_list in splits.items():
            for img_path in img_list:
                img_data = load_and_preprocess_image(img_path)
                
                if img_data is not None:
                    img_name = os.path.basename(img_path)
                    dst_path = os.path.join(PROCESSED_PATH, split_name, dst_cls, img_name)
                    # Convert float back to uint8 for saving logic, training generator will re-normalize
                    cv2.imwrite(dst_path, (img_data * 255).astype(np.uint8))
                    
    print("Dataset split and resize complete.")
    return True

def apply_augmentation():
    print("Applying augmentation to training data...")
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # We will use this generator directly in training rather than expanding disk size.
    print("Augmentation strategy defined (will be applied on-the-fly during training).")

if __name__ == '__main__':
    print("Starting dataset preparation...")
    # 1. Prepare Folder Structure
    prepare_directories()
    # 2. Resize, Normalize, and Split
    success = split_and_save_data()
    if success:
        # 3. Augmentation setup statement
        apply_augmentation()
        print(f"Data ready at {PROCESSED_PATH}")
