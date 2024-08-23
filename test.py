from flask import Flask, request, send_file
from PIL import Image
import io
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
import numpy as np
app = Flask(__name__)


# Define the output channels
OUTPUT_CHANNELS = 3

# Rebuild the models using the same architecture
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')



# Load the weights from the .h5 file for each model
generator_g.load_weights('generate_after.h5')



# Initialize the optimizers with the same settings
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# Optionally, compile the models if you plan to continue training
generator_g.compile(optimizer=generator_g_optimizer)



def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image



# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image




def process_image(image):
    image = random_jitter(image)
    image = normalize(image)
    
    return image



BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

output_signature = tf.TensorSpec(shape=(), dtype=tf.string)



@app.route('/process-image', methods=['POST'])
def predict_image():
 
    if 'image' not in request.files:
        return "No image file found in request", 400
    
    img_file = request.files['image']
    image = Image.open(img_file)
    image = tf.io.decode_image(image,channels=3)
    image = tf.ensure_shape(image, [None, None, 3])
    image = tf.cast(image, tf.float32)
    preprocessed_img = process_image(image)
    prediction = generator_g(preprocessed_img)
    prediction_image=prediction[0] * 0.5 + 0.5

    img_io = io.BytesIO()
    prediction_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


@app.route('/', methods=['GET'])
def base():
    return "hello world"

if __name__ == '__main__':
    app.run(debug=True)
