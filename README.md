# ChAd Detector (for child-adult detector)
## An Image Classification API using VGG16 Model
This project is a prototype API that predicts the class of an image as either 'adult' or 'child' using a pre-trained VGG16 convolutional neural network model. The model was trained on a dataset of images containing adults and children using transfer learning.

The project has two main parts:

- Training the VGG16 model: The model was trained using a dataset of images containing adults and children. Data augmentation techniques were applied to increase the size of the dataset and reduce overfitting. The pre-trained VGG16 model was used as a base model and fine-tuned for the specific task of classifying adult and child images.

- API development: An API was developed using Flask to make predictions on images. Two routes were defined, one that takes an image URL and another that takes a base64 encoded image. The API loads the trained VGG16 model and predicts the class of the image. The output of the API is a JSON object containing the predicted class.

## Dependencies
The following libraries are used in this project:

- numpy
- pandas
- matplotlib
- seaborn
- tensorflow
- keras
- flask
- flask_cors
- base64
- requests
- io

## Files

- **train.py**: This script trains the VGG16 model on the provided dataset of images containing adults and children. It applies data augmentation techniques to increase the size of the dataset and reduce overfitting. The trained model is saved in vgg_model.h5.
- **app.py**: This script defines the API and loads the trained VGG16 model to make predictions on images. There are two routes defined in the API, one that takes an image URL and another that takes a base64 encoded image.
- **data**: This directory contains the dataset of images containing adults and children. It has two subdirectories, train and test, each containing subdirectories for adult and child images.
- **templates**: This directory contains HTML files for displaying the output of the API.
- **static**: This directory contains CSS and JavaScript files for styling the HTML files.

## Usage
### Training the VGG16 Model
To train the VGG16 model, run the following command:

```sh
python train.py
```

### Running the API
To run the API, run the following command:

```sh
python app.py
```

The API will be available at http://127.0.0.1:5000/.

There are two routes defined in the API:

- /api/v1/resources/images?url=<image_url>: This route takes an image URL as a parameter and returns the predicted class of the image as a JSON object.

- /predict_base64: This route takes a base64 encoded image in the request body and returns the predicted class of the image as a JSON object.


## Examples
Using the /api/v1/resources/images route

To predict the class of an image using the url parameter, use the following format:

```sh
http://127.0.0.1:5000/api/v1/resources/images?url=<image_url>
```

Example:

```sh
http://127.0.0.1:5000/api/v1/resources/images?url=https://www.example.com/image.jpg
```

Using the /predict_base64 route

To predict the class of an image using a base64 encoded image in the request body, use the following command:

```sh
curl -d '{"image": "<base64_encoded_image>"}' -H "Content-Type: application/json" -X
```
