import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# 1. Load the saved model
model = tf.keras.models.load_model('change_detection_model.h5')

# 2. Define preprocessing function
def preprocess_images(image_a_path, image_b_path, img_size=256):
    """Load and preprocess image pairs for prediction"""
    # Read images
    img_a = Image.open(image_a_path).convert('RGB')
    img_b = Image.open(image_b_path).convert('RGB')
    
    # Resize
    img_a = img_a.resize((img_size, img_size))
    img_b = img_b.resize((img_size, img_size))
    
    # Convert to numpy arrays and normalize
    img_a = np.array(img_a) / 255.0
    img_b = np.array(img_b) / 255.0
    
    # Add batch dimension
    img_a = np.expand_dims(img_a, axis=0)
    img_b = np.expand_dims(img_b, axis=0)
    
    return img_a, img_b

# 3. Prediction function
def detect_changes(image_a_path, image_b_path, threshold=0.5):
    """
    Detect changes between two images
    Args:
        threshold: Confidence threshold for change pixels (0-1)
    Returns:
        change_mask: Binary mask (1=changed, 0=unchanged)
        change_percentage: % of changed pixels
    """
    # Preprocess
    img_a, img_b = preprocess_images(image_a_path, image_b_path)
    
    # Predict
    pred = model.predict([img_a, img_b])
    change_mask = (pred[0,...,0] > threshold).astype(np.uint8)
    
    # Calculate change percentage
    change_percentage = 100 * np.mean(change_mask)
    
    return change_mask, change_percentage

# 4. Visualization function
def visualize_results(image_a_path, image_b_path, change_mask):
    """Display input images and change detection results"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show Image A
    ax1.imshow(Image.open(image_a_path))
    ax1.set_title('Image A (Before)')
    ax1.axis('off')
    
    # Show Image B
    ax2.imshow(Image.open(image_b_path))
    ax2.set_title('Image B (After)')
    ax2.axis('off')
    
    # Show Change Mask
    ax3.imshow(change_mask, cmap='gray')
    ax3.set_title(f'Change Detection (White=Changed)\n{change_percentage:.2f}% changed')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

# 5. Main execution
if __name__ == "__main__":
    # Paths to your test images
    image_a_path = r"C:\ml\LEVIR-CD+\test\A\train_651.png"  # Replace with your path
    image_b_path = r"C:\ml\LEVIR-CD+\test\B\train_651.png"  # Replace with your path
    
    # Detect changes
    change_mask, change_percentage = detect_changes(image_a_path, image_b_path)
    
    # Visualize results
    visualize_results(image_a_path, image_b_path, change_mask)
    
    # Save the change mask
    Image.fromarray(change_mask * 255).save("change_mask.png")
    print(f"Saved change mask to 'change_mask.png'")