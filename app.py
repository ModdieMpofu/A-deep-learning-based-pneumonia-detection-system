from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Creating a Flask Instance
app = Flask(__name__)

# Configuration
IMAGE_SIZE = (224, 224)  # Teachable Machine model's expected input size
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Loading Pre-trained Model ...")
model = load_model('keras_model.h5', compile=False)

def image_preprocessor(path):
    '''
    Function to pre-process the image before feeding to the model.
    '''
    print('Processing Image ...')
    image = Image.open(path).convert("RGB")
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # Load the image into an array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

def model_pred(image):
    '''
    Performs predictions based on the input image
    '''
    print("Image_shape", image.shape)
    print("Image_dimension", image.ndim)
    # Returns class probabilities:
    prediction_probs = model.predict(image)[0]
    print("Prediction probabilities:", prediction_probs)
    # Get the class with the highest probability
    predicted_class = np.argmax(prediction_probs)
    confidence_score = prediction_probs[predicted_class]
    return predicted_class, confidence_score

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    # Checks if a post request was submitted
    if request.method == 'POST':
        # check if the post request has the file part
        if 'imageFile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        # check if filename is an empty string
        file = request.files['imageFile']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # if file is uploaded
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            imgPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(imgPath)
            print(f"Image saved at {imgPath}")
            # Preprocessing Image
            image = image_preprocessor(imgPath)
            # Performing Prediction
            pred_class, confidence_score = model_pred(image)
            print(f"Predicted class: {pred_class}, Confidence score: {confidence_score}")

            if pred_class == 0:
                result = 0
            elif pred_class == 1:
                result = 1
            else:
                result = "Unknown"

            return render_template('upload.html', name=filename, result=result, confidence=confidence_score)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
