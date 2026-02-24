from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")

model = tf.keras.models.load_model("../models/digit_model.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    image_data = base64.b64decode(data.split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("L")
    image = image.resize((28,28))
    image = np.array(image) / 255.0
    image = 1 - image
    image = image.reshape(1,28,28)

    prediction = model.predict(image)
    digit = int(np.argmax(prediction))

    return jsonify({"digit": digit})

if __name__ == "__main__":
    app.run(debug=True)
