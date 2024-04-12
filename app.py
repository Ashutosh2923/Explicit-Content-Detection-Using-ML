from flask import Flask, render_template, request
from tensorflow import keras
import numpy as np
from PIL import Image
import io

model = keras.models.load_model('C:\\Users\\Mayank Salakke\\Downloads\\PBL website (done)\\PBL website (done)\\PBL website\\model.h5')
height, width = 224, 224  # Replace with the dimensions expected by your model

app = Flask(__name__)

def process_image(image_data):
    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data))
    
    # Resize the image to the required input shape (height, width)
    target_size = (height, width)
    image = image.resize(target_size)
    
    # Convert the image to a NumPy array and normalize the pixel values
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    
    # Add an extra dimension to match the model's input shape (height, width, channels)
    processed_image = np.expand_dims(image_array, axis=0)
    
    return processed_image

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/check', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('home.html', prediction='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('home.html', prediction='No selected file')

    if file:
        image_data = file.read()
        arr = process_image(image_data)  # Call the process_image function here
        pred = model.predict(arr)
        # Assuming pred is a categorical prediction, get the class with the highest probability
        predicted_class = np.argmax(pred)
        # You can convert the predicted_class to a human-readable label if needed
        # For example: predicted_label = get_label_from_class(predicted_class)
        return render_template('after.html', prediction=predicted_class)

if __name__ == "__main__":
    app.run(debug=True)
