from django.shortcuts import render
import tensorflow
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = tensorflow.keras.models.load_model("models/cotton/cotton.keras")


# Create your views here.
def index(request):
    crop_image = request.files.get("crop_image")
    index_to_class = {
        0: 'Aphids',
        1: 'Army worml̥',
        2: 'Bacterial Blight',
        3: 'Healthy',
        4: 'Powdery Mildew',
        5: 'Target spot'
    }
    pillowobj = Image.open(crop_image)
    pillowobj = pillowobj.resize((150, 150))
    img_array = img_to_array(pillowobj)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    result = index_to_class[predicted_class[0]]
    print(result)
    pass
