from flask import Flask, request, send_file
from PIL import Image
import io
import tensorflow as tf
import numpy as np
from tensorflow_examples.models.pix2pix import pix2pix

app = Flask(__name__)

# Constants
OUTPUT_CHANNELS = 3
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Load the pre-trained generator model
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_g.load_weights('generate_after.h5')

# Preprocessing functions
def random_jitter(image):
    # Resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Random cropping to 256 x 256 x 3
    image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    # Random mirroring
    image = tf.image.random_flip_left_right(image)
    return image

def normalize(image):
    # Normalizing the image to [-1, 1]
    image = (image / 127.5) - 1
    return image

def preprocess_image(image):
    """
    Preprocess the image before feeding it to the model.
    """
    image = random_jitter(image)
    image = normalize(image)
    return image

@app.route('/process-image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return "No image file found in request", 400
    
    # Get the image from the request
    img_file = request.files['image']
    
    # Open the image using PIL and convert it to RGB
    image = Image.open(img_file).convert('RGB')
    
    # Convert the PIL image to a TensorFlow tensor
    image = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
    
    # Preprocess the image
    preprocessed_img = preprocess_image(image)
    
    # Add a batch dimension
    preprocessed_img = tf.expand_dims(preprocessed_img, axis=0)
    
    # Make prediction
    prediction = generator_g(preprocessed_img)
    
    # Postprocess the prediction
    prediction_image = prediction[0] * 0.5 + 0.5  # Rescale to [0, 1]
    prediction_image = tf.clip_by_value(prediction_image, 0.0, 1.0)
    
    # Convert tensor to a NumPy array and then to a PIL image
    prediction_image_np = (prediction_image.numpy() * 255).astype(np.uint8)
    output_image = Image.fromarray(prediction_image_np)
    
    # Save the output image to a bytes buffer
    img_io = io.BytesIO()
    output_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Return the processed image
    return send_file(img_io, mimetype='image/png')

@app.route('/', methods=['GET'])
def base():
    return "hello world"

if __name__ == '__main__':
    app.run(debug=True)
