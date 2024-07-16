import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the saved model
model = tf.keras.models.load_model('animal_classifier_model.keras')

# Define the image size and the class labels
img_height = 180
img_width = 180
class_names = ['cat', 'dog', 'snake']

def preprocess_image(img_path):
    """Load and preprocess an image."""
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = img_array / 255.0  # Normalize the image
    return img_array

def predict_image(img_path):
    """Predict the class of a single image."""
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence

# Example usage
if __name__ == '__main__':
    # Define the path to the image you want to predict
    img_path = 'snake.jpg'  # Update to the correct path

    # Make a prediction
    predicted_class, confidence = predict_image(img_path)

    print(f"This image most likely belongs to {predicted_class} with a {confidence:.2f}% confidence.")

    # Optional: Predict on multiple images in a directory
    # Uncomment and update the directory path as needed
    # img_dir = 'C:/Users/acer/Desktop/practice/Animal classification/test_images'
    # for img_name in os.listdir(img_dir):
    #     img_path = os.path.join(img_dir, img_name)
    #     predicted_class, confidence = predict_image(img_path)
    #     print(f"{img_name}: {predicted_class} ({confidence:.2f}%)")
