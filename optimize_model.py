import tensorflow as tf
import os

MODEL_PATH = 'models/drowsiness_model.keras'
TFLITE_MODEL_PATH = 'models/drowsiness_model.tflite'
PROCESSED_PATH = 'dataset_processed/train'
IMG_SIZE = 112

def representative_data_gen():
    """ Provides a few samples of training data so the converter can calibrate quantizations. """
    # Use standard keras utility to load a few batches
    dataset = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_PATH,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        color_mode='rgb'
    )
    # We yield only 100 samples
    for data, _ in dataset.take(100):
        # We need to explicitly normalize data to [0.0, 1.0] since that's what the model expects
        yield [tf.cast(data, tf.float32) / 255.0]

def convert_to_tflite():
    print(f"Loading keras model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Converting to TFLite (Quantized - INT8)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Enforce full integer quantization for all inputs and outputs
    converter.inference_input_type = tf.float32 # Keep float32 interface for easier Flutter integration
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
        
    model_size_kb = os.path.getsize(TFLITE_MODEL_PATH) / 1024
    print(f"TFLite Model saved to: {TFLITE_MODEL_PATH}")
    print(f"Optimized Model Size: {model_size_kb:.2f} KB ({model_size_kb/1024:.2f} MB)")
    
    if model_size_kb / 1024 < 10:
        print("Success: Model is under the 10MB limit limit.")
    else:
        print("Warning: Model size exceeds 10MB.")

if __name__ == "__main__":
    convert_to_tflite()
