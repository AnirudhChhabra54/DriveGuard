import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PROCESSED_PATH = 'dataset_processed'
IMG_SIZE = 112
BATCH_SIZE = 32

if not os.path.exists('models/drowsiness_model.keras'):
    print("Model not found!")
else:
    model = tf.keras.models.load_model('models/drowsiness_model.keras')
    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow_from_directory(
        os.path.join(PROCESSED_PATH, 'validation'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    test_generator = val_datagen.flow_from_directory(
        os.path.join(PROCESSED_PATH, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    print("Evaluating on Validation Set:")
    val_loss, val_acc = model.evaluate(val_generator, verbose=0)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    
    print("Evaluating on Test Set:")
    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
