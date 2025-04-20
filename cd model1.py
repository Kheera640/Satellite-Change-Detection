import os
import re
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/lib/x64")

import tensorflow as tf
from tensorflow.keras import layers, models, applications
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = "C:/ml/LEVIR-CD+"
IMG_SIZE = 256
BATCH_SIZE = 4

# --------------------------
# Model Building
# --------------------------
def build_model():
    # Shared Encoder (ResNet50)
    base_model = applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = True

    # Inputs
    input_a = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    input_b = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Feature extraction
    features_a = base_model(input_a)
    features_b = base_model(input_b)

    # Feature difference
    diff = layers.Subtract()([features_a, features_b])
    diff = layers.Activation("relu")(diff)

    # Decoder
    x = layers.Conv2DTranspose(512, 3, strides=2, padding="same")(diff)  # 8→16
    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)    # 16→32
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)    # 32→64
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)     # 64→128
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)     # 128→256
    
    # Output
    outputs = layers.Conv2D(1 , 1, activation="sigmoid")(x)

    return models.Model(inputs=[input_a, input_b], outputs=outputs)

# --------------------------
# Data Pipeline
# --------------------------
def load_and_preprocess(image_a_path, image_b_path, mask_path):
    # Read and process images
    image_a = tf.io.read_file(image_a_path)
    image_a = tf.image.decode_png(image_a, channels=3)
    image_a = tf.image.resize(image_a, [IMG_SIZE, IMG_SIZE])
    image_a = tf.image.convert_image_dtype(image_a, tf.float32)
    
    image_b = tf.io.read_file(image_b_path)
    image_b = tf.image.decode_png(image_b, channels=3)
    image_b = tf.image.resize(image_b, [IMG_SIZE, IMG_SIZE])
    image_b = tf.image.convert_image_dtype(image_b, tf.float32)
    
    # Process mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method='nearest')
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    
    return (image_a, image_b), mask

def augment(images, mask):
    img_a, img_b = images
    
    # Random flips
    if tf.random.uniform(()) > 0.5:
        img_a = tf.image.flip_left_right(img_a)
        img_b = tf.image.flip_left_right(img_b)
        mask = tf.image.flip_left_right(mask)
    
    if tf.random.uniform(()) > 0.5:
        img_a = tf.image.flip_up_down(img_a)
        img_b = tf.image.flip_up_down(img_b)
        mask = tf.image.flip_up_down(mask)
    
    # Random rotation
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    img_a = tf.image.rot90(img_a, k)
    img_b = tf.image.rot90(img_b, k)
    mask = tf.image.rot90(mask, k)
    
    # Color augmentations
    img_a = tf.image.random_brightness(img_a, 0.15)
    img_a = tf.image.random_contrast(img_a, 0.8, 1.2)
    img_b = tf.image.random_brightness(img_b, 0.15)
    img_b = tf.image.random_contrast(img_b, 0.8, 1.2)
    
    # Clip values
    img_a = tf.clip_by_value(img_a, 0.0, 1.0)
    img_b = tf.clip_by_value(img_b, 0.0, 1.0)
    
    return (img_a, img_b), mask

def create_dataset(split="train"):
    base_path = os.path.join(DATA_PATH, split)
    
    # File sorting
    def sort_key(x):
        numbers = re.findall(r'\d+', x)
        return int(numbers[0]) if numbers else 0
    
    a_files = sorted(
        [f for f in os.listdir(os.path.join(base_path, "A")) if f.endswith(".png")],
        key=sort_key
    )
    b_files = sorted(
        [f for f in os.listdir(os.path.join(base_path, "B")) if f.endswith(".png")],
        key=sort_key
    )
    mask_files = sorted(
        [f for f in os.listdir(os.path.join(base_path, "label")) if f.endswith(".png")],
        key=sort_key
    )

    # Verify alignment
    assert len(a_files) == len(b_files) == len(mask_files), "Mismatched file counts"
    assert all(a == b == m for a, b, m in zip(a_files, b_files, mask_files)), "Filename mismatch"

    # Create dataset
    a_paths = [os.path.join(base_path, "A", f) for f in a_files]
    b_paths = [os.path.join(base_path, "B", f) for f in b_files]
    mask_paths = [os.path.join(base_path, "label", f) for f in mask_files]

    dataset = tf.data.Dataset.from_tensor_slices((a_paths, b_paths, mask_paths))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if split == "train":
        dataset = dataset.map(augment).shuffle(100)
        
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Initialize model
    model = build_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy'
        ]
    )
    
    # Create datasets
    train_dataset = create_dataset("train")
    test_dataset = create_dataset("test")
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=1
    )
    model.save("C:/ml/change_detection_model.h5")
    
