# app.py
import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = load_model('mnist_cnn_model.h5')

def preprocess_image(image_path):
    from PIL import Image, ImageOps
    img = Image.open(image_path).convert('L')  # grayscale
    img = ImageOps.invert(img)                # putih di atas hitam
    img = img.resize((28, 28))                # resize ke ukuran MNIST
    img_array = np.array(img) / 255.0         # normalisasi
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = preprocess_image(filepath)
            pred = model.predict(img)
            prediction = f"Hasil Prediksi: {np.argmax(pred)}"

    return render_template('index.html', prediction=prediction, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
