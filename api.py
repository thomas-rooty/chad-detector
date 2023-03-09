import base64

import re
import flask
from flask import request, jsonify, send_file
from flask_cors import CORS, cross_origin
from PIL import Image
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers, losses
from pathlib import Path
import os.path
import itertools
from io import BytesIO
import requests
import keras_ocr

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()


# Make an api that uses the vgg_model.h5 file to predict the class of an image (child or adult)
# The api should take in an url of an image and return the class of the image

# Use flask
app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)


# Load model
model = keras.models.load_model('vgg_model.h5')

@app.route('/image')
def get_image():
    return send_file('fichier.jpeg', mimetype='image/jpeg')

# Define api
@app.route('/api', methods=['GET'])
def home():
  return '''<h1>API</h1>
<p>A prototype API for predicting the class of an image.</p>'''

# A route to return the class of an image
@app.route('/api/v1/resources/images', methods=['GET'])
def api_id():
  # Check if an image url was provided as part of the URL.
  # If image url is provided, assign it to a variable.
  # If no image url is provided, display an error in the browser.
  if 'url' in request.args:
    url = str(request.args['url'])
    print(url)
  else:
    return "Error: No url field provided. Please specify an image url."

  # Download image from URL and save to file
  response = requests.get(url)
  with open('temp_img.jpg', 'wb') as f:
    f.write(response.content)

  # Load image
  img = image.load_img('temp_img.jpg', target_size=(150, 150))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = img / 255

  # Predict class
  pred = model.predict(img)
  pred = np.argmax(pred, axis=1)
  pred = pred.tolist()

  # Return class knowing if 0 is adult and 1 is child as array class: adult or class: child
  if pred[0] == 0:
    return jsonify({'class': 'adult'})
  else:
    return jsonify({'class': 'child'})

# Prediction API route using image base64
@app.route('/predict_base64', methods=['POST'])
def predict():
    # Get image from url post
    input_json = request.get_json(force=True)
    res = {'image': input_json['image']}

    # prepare res['image'] to be converted to BytesIO
    res['image'] = res['image'].split(',')[1]
    res['image'] = base64.b64decode(res['image'])

    # Convert res['image'] to BytesIO
    imgmodel = BytesIO(res['image'])

    # Load image
    img = image.load_img(imgmodel, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255

    # Predict class
    pred = model.predict(img)
    pred = np.argmax(pred, axis=1)
    pred = pred.tolist()

    # Return class knowing if 0 is adult and 1 is child as array class: adult or class: child
    if pred[0] == 0:
        return jsonify({'class': 'adult'})
    else:
        return jsonify({'class': 'child'})

@app.route('/predict_base64_ocr', methods=['POST'])
def predict_ocr():

    # get image from url post
    input_json = request.get_json(force=True)
    res = {'image': input_json['image']}

    img = res["image"]
    base64_string = img.replace("data:image/jpeg;base64,", "")
    img_data = base64.b64decode(base64_string)

    img = Image.open(BytesIO(img_data))

    img = img.resize((1920, 1920))

    # Enregistrer l'image en tant que fichier jpeg nommé "fichier.jpeg"
    img.save("fichier.jpeg", "JPEG")

    images = [
        keras_ocr.tools.read(url) for url in [
            'http://localhost:5000/image'
        ]
    ]

    prediction_groups = pipeline.recognize(images)

    predicted_image_1 = prediction_groups[0]

    textList = ""
    for text, box in predicted_image_1:
        textList += text + " "

    # make a regex for find 4digits continue
    regex_year = r"\b(\d{4})\b"

    resultats = re.findall(regex_year, textList)

    date = resultats[0]

    return jsonify({'date': date})


app.run()

if __name__ == '__main__':
  app.run()
