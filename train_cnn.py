import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# Configuration
PROCESSED_PATH = 'dataset_processed'
IMG_SIZE = 112
BATCH_SIZE = 32
EPOCHS = 25

def build_model():
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 4 (Optional based on performance)
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        # Binary Classification output: Open/Closed
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def get_data_generators():
    # Only training data gets augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation data is only read directly
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_PATH, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(PROCESSED_PATH, 'validation'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    return train_generator, val_generator

def plot_training(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    plt.savefig('training_plots.png')
    print("Saved training progress plot to training_plots.png")

def train():
    os.makedirs('models', exist_ok=True)
    
    print("Initializing Generators...")
    train_gen, val_gen = get_data_generators()
    
    print("Class Indices mapping:", train_gen.class_indices)
    
    model = build_model()
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        filepath='models/drowsiness_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint]
    )
    
    plot_training(history)
    print("Training complete! Model saved to models/drowsiness_model.keras")

if __name__ == "__main__":
    train()
